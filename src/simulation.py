from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import gc

from tinygrad import Tensor, Device, dtypes, TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from tinygrad.nn.optim import AdamW

from model.agents import AgentModel, ConvConfig

COLORS = {
    "WALL": (0, 0, 0),         # Black - Impassable
    "FLOOR": (255, 255, 255), # White - Walkable
    "TRAP": (255, 0, 0),      # Red - Instant kill
    "WIND": (0, 0, 255),      # Blue - Drains energy
    "POWER_CELL": (0, 255, 0) # Green - Replenishes energy
}

class MazeMapGenerator:
    """
    Generates a procedurally created maze-like grid world.
    Ensures that traps and other objects do not block paths.
    """
    def __init__(self, width: int, height: int):
        if width % 2 == 0: width += 1
        if height % 2 == 0: height += 1
        self.width = width
        self.height = height
        self.map_grid = None

    def _initialize_grid(self):
        self.map_grid = np.full((self.height, self.width, 3), COLORS["WALL"], dtype=np.uint8)

    def _is_valid(self, y, x):
        return 0 <= y < self.height and 0 <= x < self.width

    def _prim_maze_generation(self):
        start_y, start_x = (random.randint(0, self.height // 2) * 2 + 1,
                            random.randint(0, self.width // 2) * 2 + 1)
        self.map_grid[start_y, start_x] = COLORS["FLOOR"]
        frontier = []
        for dy, dx in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            ny, nx = start_y + dy, start_x + dx
            if self._is_valid(ny, nx): frontier.append((ny, nx))
        while frontier:
            fy, fx = random.choice(frontier); frontier.remove((fy, fx))
            neighbors = []
            for dy, dx in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                ny, nx = fy + dy, fx + dx
                if self._is_valid(ny, nx) and np.array_equal(self.map_grid[ny, nx], COLORS["FLOOR"]):
                    neighbors.append((ny, nx))
            if neighbors:
                ny, nx = random.choice(neighbors)
                self.map_grid[fy, fx] = COLORS["FLOOR"]
                self.map_grid[(fy + ny) // 2, (fx + nx) // 2] = COLORS["FLOOR"]
                for dy, dx in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                    nfy, nfx = fy + dy, fx + dx
                    if self._is_valid(nfy, nfx) and np.array_equal(self.map_grid[nfy, nfx], COLORS["WALL"]):
                        if (nfy, nfx) not in frontier: frontier.append((nfy, nfx))

    def _add_rooms(self, num_rooms, min_size=3, max_size=7):
        for _ in range(num_rooms):
            room_w = random.randrange(min_size, max_size + 1, 2)
            room_h = random.randrange(min_size, max_size + 1, 2)
            room_x = random.randrange(1, self.width - room_w, 2)
            room_y = random.randrange(1, self.height - room_h, 2)
            self.map_grid[room_y:room_y+room_h, room_x:room_x+room_w] = COLORS["FLOOR"]

    def _get_available_floor_tiles(self):
        return np.argwhere(np.all(self.map_grid == COLORS["FLOOR"], axis=-1))

    def _is_safe_to_replace(self, y: int, x: int) -> bool:
        floor_neighbors = 0
        up = (y-1, x)
        down = (y+1, x)
        left = (y, x-1)
        right = (y, x+1)
        
        is_up_floor = self._is_valid(*up) and not np.array_equal(self.map_grid[up], COLORS["WALL"])
        is_down_floor = self._is_valid(*down) and not np.array_equal(self.map_grid[down], COLORS["WALL"])
        is_left_floor = self._is_valid(*left) and not np.array_equal(self.map_grid[left], COLORS["WALL"])
        is_right_floor = self._is_valid(*right) and not np.array_equal(self.map_grid[right], COLORS["WALL"])

        # Check for horizontal and vertical corridors
        if is_up_floor and is_down_floor and not is_left_floor and not is_right_floor: return False
        if not is_up_floor and not is_down_floor and is_left_floor and is_right_floor: return False
        
        return True

    def _place_stationary_objects(self, trap_ratio: float, wind_ratio: float):
        all_floor_tiles = self._get_available_floor_tiles()
        safe_floor_tiles = [
            (y, x) for y, x in all_floor_tiles if self._is_safe_to_replace(y, x)
        ]
        random.shuffle(safe_floor_tiles)
        num_safe_tiles = len(safe_floor_tiles)
        num_traps = int(num_safe_tiles * trap_ratio)
        num_winds = int(num_safe_tiles * wind_ratio)
        
        for _ in range(num_traps):
            if not safe_floor_tiles: break
            y, x = safe_floor_tiles.pop()
            self.map_grid[y, x] = COLORS["TRAP"]
        for _ in range(num_winds):
            if not safe_floor_tiles: break
            y, x = safe_floor_tiles.pop()
            self.map_grid[y, x] = COLORS["WIND"]

    def generate_new_map(self, num_rooms=15, trap_ratio=0.03, wind_ratio=0.05):
        self._initialize_grid()
        self._prim_maze_generation()
        self._add_rooms(num_rooms)
        self.map_grid[0, :] = self.map_grid[-1, :] = COLORS["WALL"]
        self.map_grid[:, 0] = self.map_grid[:, -1] = COLORS["WALL"]
        self._place_stationary_objects(trap_ratio, wind_ratio)
        print(f"Generated a {self.width}x{self.height} maze.")
        return self.map_grid.copy()

class Agent:
    def __init__(self, agent_id: int, pos: tuple, initial_tile_color: tuple, config: ConvConfig, initial_weights=None):
        self.id = agent_id
        self.y, self.x = pos
        self.energy = 100
        self.age = 0
        self.is_alive = True

        self.start_epsilon = 0.8  # Start with 80% chance of random action
        self.end_epsilon = 0.1    # Decay to 10% chance of random action
        self.epsilon_decay_steps = 25 # The number of steps over which to decay
        
        self.config = config
        self.model = AgentModel(config)
        if initial_weights:
            load_state_dict(self.model, initial_weights)

            for k, v in get_state_dict(self.model).items():  
                v.to_(Device.DEFAULT)
        self.params = get_parameters(self.model)
        for p in self.params:
            p.requires_grad = True
        self.optimizer = AdamW(self.params, lr=1e-4)
        
        # 0:up, 1:down, 2:left, 3:right, 4:idle
        self.action_history = deque([4] * 63, maxlen=63) 
        self.last_pain_signal = self.calculate_pain_signal(initial_tile_color, None, hit_a_wall=False)

        r, g, b = 255, random.randint(100, 200), 0
        self.color = (r, g, b)

    def get_perception(self, world_map: np.ndarray) -> Tensor:
        """Extracts the 9x9 view around the agent."""
        h, w, _ = world_map.shape
        view = np.full((9, 9, 3), COLORS["WALL"], dtype=np.uint8) # Pad with walls
        
        y_start, x_start = self.y - 4, self.x - 4
        y_end, x_end = self.y + 5, self.x + 5
        
        # Calculate slices for world and view to handle boundaries
        world_y_slice = slice(max(0, y_start), min(h, y_end))
        world_x_slice = slice(max(0, x_start), min(w, x_end))
        view_y_slice = slice(max(0, -y_start), 9 - max(0, y_end - h))
        view_x_slice = slice(max(0, -x_start), 9 - max(0, x_end - w))

        view[view_y_slice, view_x_slice] = world_map[world_y_slice, world_x_slice]
        
        # Normalize to [0, 1] and convert to Tensor
        # (C, H, W) format expected by Conv2d
        perception_tensor = Tensor(view.astype(np.float32) / 255.0, device=Device.DEFAULT).permute(2, 0, 1).unsqueeze(0)
        return perception_tensor

    def calculate_pain_signal(self, current_tile_color: tuple, last_action_idx: int, hit_a_wall: bool) -> float:
        """Calculates the composite pain signal based on internal state."""
        # Weighting factors
        w_energy = 2.0
        w_comp = 0.5
        
        # 1. Energy Pain (inverse relationship, normalized)
        # Pain is high when energy is low.
        pain_energy = (1.0 - (self.energy / 100.0))
        
        # 2. Computational Load / Effort Pain
        pain_comp = 0.0
        if hit_a_wall:
            pain_comp = 1.0  # High cost for an unproductive, invalid move
        elif current_tile_color == COLORS["WIND"]:
            pain_comp = 1.0 # High cost in a wind tile
        elif last_action_idx is not None and last_action_idx != 4: # Moving
            pain_comp = 0.5 # Medium cost for moving
        else: # Idle
            pain_comp = 0.1 # Low cost for idling
        
        return (w_energy * pain_energy) + (w_comp * pain_comp)
    
    @TinyJit
    def _inference(self, perception_tensor: Tensor, memory_tensor: Tensor) -> Tensor:

        output = self.model(perception_tensor, memory_tensor)
        logits = output["logits"].realize()
        return logits

    def choose_action(self, perception_tensor: Tensor) -> int:
        """Uses the model to decide the next action with decaying epsilon-greedy exploration."""
        # Calculate the progress of the decay, clamped between 0.0 and 1.0
        progress = min(1.0, self.age / self.epsilon_decay_steps)
        # Linearly interpolate from start_epsilon to end_epsilon based on progress
        current_epsilon = self.start_epsilon - (self.start_epsilon - self.end_epsilon) * progress

        if random.random() < current_epsilon:
            # EXPLORE: Choose a random action
            action_idx = random.randint(0, self.config.vocab_size - 1)
        else:
            val_np = np.random.uniform(low=0.0, high=1.0)
            t = Tensor(val_np)
            memory_tensor = Tensor([list(self.action_history)], device=Device.DEFAULT, dtype=dtypes.int32)
            logits = self._inference(perception_tensor, memory_tensor)

            action_logits = logits[0, -1, :]

            probs = (action_logits / 1.0).softmax()
            action_idx = (probs.cumsum() > t.item()).argmax().numpy().item()

        return action_idx

    @TinyJit
    def learn(self, current_tile_color: tuple, perception_tensor: Tensor, last_perception_tensor: Tensor, action_idx: int, hit_a_wall: bool):
        """Intra-life learning step to minimize pain. (Corrected with graph-safe conditional)"""
        with Tensor.train():
            self.optimizer.zero_grad()
            
            # Re-create the memory tensor from the state when the action was taken.
            # Note: The *last* action in this deque is the one we are evaluating.
            memory_tensor = Tensor([list(self.action_history)], device=Device.DEFAULT, dtype=dtypes.int32)
            
            # --- Forward pass from the PREVIOUS state to get the prediction and logits that led to the CURRENT state ---
            output = self.model(last_perception_tensor, memory_tensor)
            logits = output["logits"]
            tile_prediction = output["prediction"]
            
            # --- 1. Uncertainty Loss (Supervised) ---
            # How well did the agent predict the current view from its last view?
            loss_uncertainty = (tile_prediction - perception_tensor).square().mean()

            # --- 2. Action Loss (Reinforcement) ---
            # The agent is rewarded for actions that reduce its pain signal.
            current_pain = self.calculate_pain_signal(current_tile_color, action_idx, hit_a_wall)
            pain_delta = current_pain - self.last_pain_signal
            
            # We only care about the logit for the action we just took.
            action_logits = logits[0, -1, :]
            
            # Calculate the cross-entropy loss for the action that was actually taken.
            ce_loss = action_logits.sparse_categorical_crossentropy(Tensor([action_idx], device=Device.DEFAULT))
            
            # Use Tensor.where to create a graph-safe conditional weight.
            condition = Tensor([pain_delta], device=Device.DEFAULT) < -0.01
            
            # If pain decreased, the weight is 1.0 (learn from this good move). Otherwise, it's 0.0.
            loss_weight = condition.where(1.0, 0.0)
            
            loss_action = (ce_loss * loss_weight).sum() # Use the graph-safe weight

            # --- Combine losses and backpropagate ---
            total_loss = loss_uncertainty + loss_action
            
            total_loss.backward()
            self.optimizer.step()
            
            # Update the pain signal for the next step's comparison
            self.last_pain_signal = current_pain

    def move(self, dy: int, dx: int):
        self.y += dy
        self.x += dx

class World:
    def __init__(self, width: int, height: int, num_agents: int, num_power_cells: int):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.num_power_cells = num_power_cells
        
        self.config = ConvConfig(vocab_size=5) # 5 actions
        
        self.agents = {} # Now a dictionary
        self.next_agent_id = 0
        self._agent_positions = set()

        self.oldest_agent_ever = None
        self.oldest_agent_ever_age = -1
        
        print("--- Initializing World ---")
        self.map_generator = MazeMapGenerator(width, height)
        self.static_map = self.map_generator.generate_new_map(
            num_rooms=64, trap_ratio=0.05, wind_ratio=0.15
        )
        self.current_map = self.static_map.copy()
        
        self.available_spawn_points = self._get_floor_tiles()
        
        for _ in range(self.num_agents):
            self.spawn_agent() # Will spawn with random initial weights

        self.spawn_power_cells()

    def _get_floor_tiles(self):
        return np.argwhere(np.all(self.static_map == COLORS["FLOOR"], axis=-1)).tolist()

    def spawn_power_cells(self):
        # Clear old power cells
        self.current_map[np.all(self.current_map == COLORS["POWER_CELL"], axis=-1)] = COLORS["FLOOR"]
        
        floor_tiles = self.available_spawn_points.copy()
        # Avoid spawning on agents
        occupied_tiles = [tuple(pos) for pos in self._agent_positions]
        spawnable_tiles = [tile for tile in floor_tiles if tuple(tile) not in occupied_tiles]
        
        random.shuffle(spawnable_tiles)
        
        num_to_spawn = min(self.num_power_cells, len(spawnable_tiles))
        for i in range(num_to_spawn):
            y, x = spawnable_tiles[i]
            self.current_map[y, x] = COLORS["POWER_CELL"]
        print(f"Spawned {num_to_spawn} new power cells.")

    def spawn_agent(self, parent_weights=None):
        if not self.available_spawn_points: return
        
        spawn_pos = tuple(random.choice(self.available_spawn_points))
        while spawn_pos in self._agent_positions:
            spawn_pos = tuple(random.choice(self.available_spawn_points))

        child_weights = None
        if parent_weights:
            child_weights = {}
            for name, tensor in parent_weights.items():
                noise_np = np.random.randn(*tensor.shape).astype(np.float32) * 0.01   # σ = 0.01

                mutated_np = tensor.numpy() + noise_np                # tensor + noise

                child_weights[name] = Tensor(mutated_np, device="CPU").realize()
        
        initial_tile_color = tuple(self.static_map[spawn_pos])
        
        agent = Agent(self.next_agent_id, spawn_pos, initial_tile_color, self.config, initial_weights=child_weights)
        
        self.agents[self.next_agent_id] = agent
        
        self._agent_positions.add(spawn_pos)
        self.next_agent_id += 1
        print(f"Spawned Agent {agent.id} at {spawn_pos}")

    def get_valid_moves(self, agent: Agent):
        valid_moves = []
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]: # Right, Left, Down, Up, Idle
            ny, nx = agent.y + dy, agent.x + dx
            # Check boundaries and if the tile is not a wall or another agent
            if (0 <= ny < self.height and 0 <= nx < self.width and
                not np.array_equal(self.current_map[ny, nx], COLORS["WALL"]) and
                (ny, nx) not in self._agent_positions):
                valid_moves.append((dy, dx))
        return valid_moves

    def step(self):
        agents_to_remove = []
        action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)} # up, down, left, right, idle
        WALL_BUMP_PENALTY = 5 # Extra energy cost for hitting a wall
        
        # --- Cache perceptions for all agents first ---
        perceptions = {
            agent_id: agent.get_perception(self.current_map) 
            for agent_id, agent in self.agents.items()
        }

        for agent_id, agent in self.agents.items():
            if not agent.is_alive: continue
            agent.age += 1
            
            last_perception = perceptions[agent_id]
            action_idx = agent.choose_action(last_perception)
            
            dy, dx = action_map.get(action_idx, (0, 0))
            
            # --- Collision Detection and Penalty Logic ---
            ny, nx = agent.y + dy, agent.x + dx
            hit_a_wall = False
            
            # Check if the INTENDED move is invalid
            if not (0 <= ny < self.height and 0 <= nx < self.width and
                    not np.array_equal(self.current_map[ny, nx], COLORS["WALL"]) and
                    (ny, nx) not in self._agent_positions):
                
                # If the intended move was not to stay idle, it's a bump
                if action_idx != 4:
                    hit_a_wall = True
                    agent.energy -= WALL_BUMP_PENALTY

                # Invalidate move, force idle
                dy, dx = 0, 0 
                # Note: We don't change action_idx, so the agent learns from its *intended* bad action

            self._agent_positions.remove((agent.y, agent.x))
            agent.move(dy, dx) # This will be (0,0) if the move was invalid
            self._agent_positions.add((agent.y, agent.x))
            agent.action_history.append(action_idx)

            tile_color = tuple(self.current_map[agent.y, agent.x])
            
            if tile_color == COLORS["TRAP"]:
                agent.is_alive = False
                print(f"Agent {agent.id} stepped on a trap at ({agent.y}, {agent.x}) and died.")
                agents_to_remove.append(agent_id)
                continue

            if tile_color == COLORS["WIND"]:
                agent.energy -= 10 # Extra energy drain
            
            if tile_color == COLORS["POWER_CELL"]:
                agent.energy += 25
                self.current_map[agent.y, agent.x] = COLORS["FLOOR"]
                print(f"Agent {agent.id} consumed a power cell.")
            
            current_perception = agent.get_perception(self.current_map)

            # Enable the now-fixed intra-agent learning, passing the collision flag
            agent.learn(tile_color, current_perception, last_perception, action_idx, hit_a_wall)
            
            agent.energy -= 1 # Base energy cost per step

            if agent.energy <= 0:
                agent.is_alive = False
                print(f"Agent {agent.id} ran out of energy and died.")
                agents_to_remove.append(agent_id)

            gc.collect()
                
        if agents_to_remove:
            surviving_agents = [a for a in self.agents.values() if a.is_alive]
            parent = None
            if surviving_agents:
                # --- TOURNAMENT SELECTION LOGIC ---
                # 1. Determine the size of the tournament (up to 3 contestants)
                #    This handles the edge case where fewer than 3 agents are alive.
                tournament_size = min(3, len(surviving_agents))
                
                # 2. Randomly select the contestants from the pool of survivors
                contestants = random.sample(surviving_agents, tournament_size)
                
                # 3. The parent is the winner of the tournament (the one with the max age)
                parent = max(contestants, key=lambda a: a.age)
                # --- END OF TOURNAMENT SELECTION LOGIC ---
            
            parent_weights = get_state_dict(parent.model) if parent else None

            for agent_id in agents_to_remove:
                dead_agent = self.agents[agent_id]
                if dead_agent.age > self.oldest_agent_ever_age:
                    self.oldest_agent_ever_age = dead_agent.age
                    self.oldest_agent_ever = dead_agent.id
                
                pos = (dead_agent.y, dead_agent.x)
                if pos in self._agent_positions: self._agent_positions.remove(pos)
                
                # Respawn a new agent, inheriting from the best survivor
                self.spawn_agent(parent_weights=parent_weights)
                del self.agents[agent_id] # Remove the old agent from the dictionary

            gc.collect()
            
        if random.random() < 0.05: # 5% chance each step to respawn all cells
            self.spawn_power_cells()

        gc.collect()

    def get_population_metrics(self):
        """Calculates and returns key metrics about the agent population."""
        live_agents = [a for a in self.agents.values() if a.is_alive]
        if not live_agents:
            return {"avg_age": 0, "max_live_age": 0, "oldest_ever": self.oldest_agent_ever}
            
        ages = [a.age for a in live_agents]
        avg_age = sum(ages) / len(ages)
        
        oldest_living_agent = max(live_agents, key=lambda a: a.age)
        max_live_age = oldest_living_agent.age
        
        return {
            "avg_age": avg_age,
            "max_live_age": max_live_age,
            "oldest_ever_id": self.oldest_agent_ever,
            "oldest_ever_age": self.oldest_agent_ever_age,
        }

    def get_render_frame(self, pixel_per_tile=50):
        """Creates a render frame with a specified pixel size per tile."""
        frame = np.kron(self.current_map, np.ones((pixel_per_tile, pixel_per_tile, 1), dtype=np.uint8))

        for agent in self.agents.values():
            if not agent.is_alive: continue
            
            y_start, x_start = agent.y * pixel_per_tile, agent.x * pixel_per_tile
            frame[y_start:y_start+pixel_per_tile, x_start:x_start+pixel_per_tile] = agent.color
        
        return frame
    
def render_episode(world: World, episode_num: int, num_steps: int, timestamps: List = [], avg_ages: List = [], max_live_ages: List = [], oldest_ever_ages: List = []):
    """
    Runs the simulation for a given number of steps and renders the output as a GIF,
    including a side plot for population age metrics.
    """
    print(f"\n--- Starting Episode {episode_num} ---")
    
    PIXEL_PER_TILE = 50
    
    # --- 1. Setup the Figure and Subplots using GridSpec for better layout control ---
    fig = plt.figure(figsize=(30, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.1)
    
    ax_maze = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    ax_maze.set_xticks([])
    ax_maze.set_yticks([])
    
    # Initialize the image plot for the maze
    im = ax_maze.imshow(world.get_render_frame(pixel_per_tile=PIXEL_PER_TILE), animated=True)
    
    # Dictionary to keep references to the text artists for agent IDs
    agent_texts = {}

    ax_plot.set_title("Population Age Metrics")
    ax_plot.set_xlabel("Timestep")
    ax_plot.set_ylabel("Age")
    # ax_plot.set_xscale("log")
    line_avg, = ax_plot.plot([], [], label='Avg Age', color='cyan')
    line_max, = ax_plot.plot([], [], label='Oldest Living', color='lime')
    line_ever, = ax_plot.plot([], [], label='Oldest Ever', color='magenta', linestyle='--')
    ax_plot.legend()
    ax_plot.grid(True, alpha=0.3)

    def update(frame_num):
        global_timestep = frame_num + (episode_num - 1) * num_steps
        print(f"    Global Timesteps: {global_timestep}")

        # --- Run one simulation step ---
        world.step()
        
        # --- 3. Update the Maze Visualization ---
        im.set_array(world.get_render_frame(pixel_per_tile=PIXEL_PER_TILE))
        
        current_agent_ids = set(world.agents.keys())
        dead_ids = set(agent_texts.keys()) - current_agent_ids
        for agent_id in dead_ids:
            agent_texts[agent_id].set_visible(False)
            del agent_texts[agent_id]

        for agent_id, agent in world.agents.items():
            if agent_id not in agent_texts:
                agent_texts[agent_id] = ax_maze.text(0, 0, "", ha='center', va='center', color='black', fontsize=8, fontweight='bold')
            
            txt = agent_texts[agent_id]
            txt.set_text(str(agent.id))
            txt.set_position((agent.x * PIXEL_PER_TILE + PIXEL_PER_TILE/2, 
                              agent.y * PIXEL_PER_TILE + PIXEL_PER_TILE/2))
            txt.set_visible(True)

        metrics = world.get_population_metrics()
        title_text = f"Timestep: {global_timestep} | Live Agents: {len(world.agents)}"
        ax_maze.set_title(title_text, fontsize=12)
        
        # --- 4. Update the Age Metrics Plot ---
        timestamps.append(global_timestep)
        avg_ages.append(metrics['avg_age'])
        max_live_ages.append(metrics['max_live_age'])
        oldest_ever_ages.append(metrics['oldest_ever_age'])
        
        # Update the data of the line objects
        line_avg.set_data(timestamps, avg_ages)
        line_max.set_data(timestamps, max_live_ages)
        line_ever.set_data(timestamps, oldest_ever_ages)
        
        # Rescale the plot axes
        ax_plot.relim()
        ax_plot.autoscale_view()

        # Print progress to console
        if (frame_num + 1) % 100 == 0:
            print(f"  Episode {episode_num}, Step {frame_num+1}/{num_steps} | "
                  f"Oldest Ever: {metrics['oldest_ever_age']}")

        # Return all animated artists
        return [im] + list(agent_texts.values()) + [line_avg, line_max, line_ever]

    # --- Create and Save the Animation ---
    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=150, blit=True, repeat=False)
    
    output_filename = f"episode_{episode_num}.gif"
    print(f"--- Saving animation to {output_filename} ---")
    ani.save(output_filename, writer='pillow', fps=10)
    plt.close(fig)
    print(f"--- Finished Episode {episode_num} ---")

    return timestamps, avg_ages, max_live_ages, oldest_ever_ages

# --- Main Simulation and Animation Setup ---
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    GRID_WIDTH = 51 
    GRID_HEIGHT = 51
    NUM_AGENTS = 20
    NUM_POWER_CELLS = 100

    world = World(width=GRID_WIDTH, height=GRID_HEIGHT, num_agents=NUM_AGENTS, num_power_cells=NUM_POWER_CELLS)

    TOTAL_SIMULATION_STEPS = 10000
    STEPS_PER_EPISODE = 1000 # This will create animations of 1000 steps each
    
    num_episodes = TOTAL_SIMULATION_STEPS // STEPS_PER_EPISODE

    timestamps, avg_ages, max_live_ages, oldest_ever_ages = [], [], [], []

    for i in range(num_episodes):
        timestamps, avg_ages, max_live_ages, oldest_ever_ages = render_episode(world, episode_num=i + 1, num_steps=STEPS_PER_EPISODE, timestamps=timestamps, avg_ages=avg_ages, max_live_ages=max_live_ages, oldest_ever_ages=oldest_ever_ages)

    print("\n--- All episodes rendered. ---")