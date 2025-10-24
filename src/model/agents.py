import numpy as np

from dataclasses import dataclass
from typing import Tuple

from tinygrad import Tensor, Device, dtypes, TinyJit
from tinygrad.nn import Conv1d, Conv2d, ConvTranspose2d, Embedding, Linear, RMSNorm
from tinygrad.nn.state import get_parameters, get_state_dict
from tinygrad.nn.optim import Adam

class Encoder:
    def __init__(self, hidden_size):
        self.c1 = Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        self.c2 = Conv2d(in_channels=4, out_channels=8, kernel_size=3)
        self.l1 = Linear(in_features=8 * 3 * 3, out_features=hidden_size)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.c1(x).silu()
        x = self.c2(x).silu()

        x = x.flatten(start_dim=1)
        x = self.l1(x)

        return x

class Decoder:
    def __init__(self, hidden_size):
        self.l1 = Linear(in_features=hidden_size, out_features=8 * 3 * 3)
        self.ct1 = ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3)
        self.ct2 = ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=3)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).silu()
        x = x.reshape(x.shape[0], 8, 3, 3)

        x = self.ct1(x).silu()
        
        x = self.ct2(x).sigmoid()
        return x

class SwiGLU:
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.w1 = Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = Linear(intermediate_size, hidden_size, bias=False)
    def __call__(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).silu() * self.w3(x))

class LFM2ConvOperator:
    def __init__(self, hidden_size: int, conv_kernel_size: int):
        self.hidden_size = hidden_size
        self.kernel_size = conv_kernel_size
        self.in_proj = Linear(hidden_size, 3 * hidden_size, bias=False)
        self.conv = Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1, # Causal padding
            groups=hidden_size,
            bias=False
        )
        self.out_proj = Linear(hidden_size, hidden_size, bias=False)
        
    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        bsz, seq_len, _ = x.shape
        B, C, x_proj = self.in_proj(x).chunk(3, dim=-1)
        x_gated = B * x_proj

        x_gated_permuted = x_gated.permute(0, 2, 1)

        conv_out = self.conv(x_gated_permuted)[:, :, :seq_len]

        conv_out = conv_out.permute(0, 2, 1)
        output = self.out_proj(C * conv_out)
        return output

@dataclass
class ConvConfig:
    vocab_size: int = 32
    hidden_size: int = 16  # n_dim
    intermediate_size: int = 40 # FFN intermediate dim, typically 2.5 * hidden_size
    num_hidden_layers: int = 6
    conv_kernel_size: int = 3
    max_position_embeddings: int = 64
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

class ConvBlock:
    def __init__(self, config: ConvConfig):
        self.operator = LFM2ConvOperator(config.hidden_size, config.conv_kernel_size)
        self.feed_forward = SwiGLU(config.hidden_size, config.intermediate_size)
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: Tensor):
        residual = hidden_states
        # Apply the convolutional operator
        normed_hidden = self.operator_norm(hidden_states)
        hidden_states = self.operator(normed_hidden)
        hidden_states = hidden_states + residual
        
        # Apply the feed-forward network
        residual = hidden_states
        normed_hidden = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(normed_hidden)
        hidden_states = hidden_states + residual
        
        return hidden_states

class ConvModel:
    def __init__(self, config: ConvConfig):
        self.layers = [ConvBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, h: Tensor):
        for i, layer in enumerate(self.layers):
            h = layer(h)
        
        return self.norm(h)

class AgentModel:
    def __init__(self, config: ConvConfig):
        self.config = config
        self.encoder = Encoder(config.hidden_size)
        self.decoder = Decoder(config.hidden_size)
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.model = ConvModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @TinyJit
    def __call__(self, input: Tensor, memory: Tensor):
        perception = self.encoder(input) # encodes perception from current surrounding 7x7 tiles
        h = self.embed_tokens(memory) # encode action histories into embedding
        bsz, seq_len, dim = h.shape

        assert bsz == 1 # only support single batch size for single agent
        assert seq_len == 31 # memory length should be 31, if action histories less than 31 the earlier is filled with idle action

        perception = perception.reshape(bsz, 1, dim)

        h = h.cat(perception, dim=1) # include encoded perception into memory

        hidden_states = self.model(h) # forward pass to compute immediate next action

        action_hidden_states = hidden_states[:, -1, :]

        prediction = self.decoder(perception + action_hidden_states) # Next 7x7 tiles prediction based on potential chosen action

        logits = self.lm_head(hidden_states)
        
        return {"logits": logits, "prediction": prediction}

if __name__ == "__main__":
    print(f"Running on device: {Device.DEFAULT}")
    
    # --- 1. Setup Model and Optimizer ---
    config = ConvConfig(
        vocab_size=5, # 5 actions: up, down, left, right, idle
        hidden_size=16,
        intermediate_size=40,
        num_hidden_layers=6,
        conv_kernel_size=3,
    )

    model = AgentModel(config)
    optimizer = Adam(get_parameters(model), lr=1e-4)

    total_params = sum(p.numel() for p in get_parameters(model))
    print(f"\n--- Model Initialized ---")
    print(f"  Total Parameters: {total_params}")

    # --- 2. Create Dummy Data ---
    # Simulates one agent's input
    batch_size = 1
    

    print("\n--- Starting Dummy Training Loop ---")

    # --- 3. Training Loop ---
    with Tensor.train():
        for i in range(10):
            
            # Visual perception: a 7x7 RGB image, normalized
            dummy_perception = Tensor.randn(batch_size, 3, 7, 7)
            
            # Action history: sequence of 31 action indices
            dummy_memory = Tensor(
                np.random.randint(0, config.vocab_size, size=(batch_size, 31)), 
                dtype=dtypes.int32
            )
            
            # Ground truth for loss calculation
            dummy_target_action = Tensor([2], dtype=dtypes.int32) # e.g., agent should move "left"
            dummy_target_perception = Tensor.randn(batch_size, 3, 7, 7) # The view agent *should* have predicted
            
            optimizer.zero_grad()
            # --- Forward pass ---
            output = model(dummy_perception, dummy_memory)
            logits = output["logits"]
            prediction = output["prediction"]

            # --- Calculate losses ---
            
            # Uncertainty loss (how well did we predict the next view?)
            loss_uncertainty = (prediction - dummy_target_perception).square().mean()

            # Action loss (did we choose the right action?)
            # We only care about the logit for the last action in the sequence
            action_logits = logits[:, -1, :]
            loss_action = action_logits.sparse_categorical_crossentropy(dummy_target_action)

            # Combine losses (both are scalars, so this is safe)
            total_loss = loss_uncertainty + loss_action
            
            # --- Backward pass and optimization ---
            total_loss.backward()
            optimizer.step()
            
            # --- Check for gradients ---
            params = get_parameters(model)
            has_grads = all(p.grad is not None for p in params)
            
            print(f"Step {i+1:02d} | Loss: {total_loss.numpy().item():.4f} | Gradients Present: {has_grads}")
            
            if not has_grads:
                print("!!! GRADIENT CHECK FAILED !!!")
                # Print which parameters are missing gradients
                for p in params:
                    if p.grad is None:
                        for name, param in get_state_dict(model).items():
                            if id(param) == id(p):
                                print(f"  - Missing grad for: {name}")
                # break # Stop if gradients are missing