from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import gc
import time
import wandb

from tinygrad import Tensor, Device, dtypes, TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from tinygrad.nn.optim import AdamW

from model.agents import AgentModel, ConvConfig

import sys
sys.setrecursionlimit(1000000)

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
        self.energy = 200
        self.age = 0
        self.is_alive = True

        self.start_epsilon = 0.8  # Start with 80% chance of random action
        self.end_epsilon = 0.1    # Decay to 10% chance of random action
        self.epsilon_decay_steps = 25 # The number of steps over which to decay
        
        self.config = config
        self.model = AgentModel(config)
        for k, v in get_state_dict(self.model).items():  
            v.realize()
        
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

        self.experience_buffer = deque(maxlen=200) 
        self.is_sleeping = False

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
        pain_energy = (1.0 - (self.energy / 200.0))
        
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
    def _learn_from_experience_jit(self, last_perceptions_batch: Tensor, current_perceptions_batch: Tensor, actions_batch: Tensor, memory_tensor: Tensor, pain_deltas_tensor: Tensor):
        with Tensor.train():
            self.optimizer.zero_grad()

            # The memory_tensor is now passed in, correctly paired with each perception
            output = self.model(last_perceptions_batch, memory_tensor)
            logits_batch = output["logits"]
            prediction_batch = output["prediction"]

            loss_uncertainty = (prediction_batch - current_perceptions_batch).square().mean()

            action_logits_batch = logits_batch[:, -1, :]
            ce_loss_batch = action_logits_batch.sparse_categorical_crossentropy(actions_batch)

            condition = pain_deltas_tensor < -0.01
            loss_weights = condition.where(1.0, 0.0)
            
            loss_action = (ce_loss_batch * loss_weights).mean()

            total_loss = loss_uncertainty + loss_action

            total_loss.backward()
            self.optimizer.step()

    def learn_from_experience(self, batch_size: int):
        """
        Intra-life learning step to minimize pain, using batched experience replay.
        This is called during the "night" or sleep phase.
        """
        if len(self.experience_buffer) < batch_size:
            return # Not enough memories to learn yet

        minibatch = random.sample(self.experience_buffer, batch_size)
        
        # UNPACK THE NEW 6-TUPLE EXPERIENCE. We no longer need tile_colors or wall_hits here.
        last_perceptions, action_indices, current_perceptions, last_pains, current_pains, histories = zip(*minibatch)

        # --- Correct Tensor Creation ---
        last_perceptions_batch = Tensor.cat(*last_perceptions, dim=0)
        current_perceptions_batch = Tensor.cat(*current_perceptions, dim=0)
        actions_batch = Tensor(list(action_indices), device=Device.DEFAULT, dtype=dtypes.int32).realize()
        memory_tensor = Tensor(list(histories), device=Device.DEFAULT, dtype=dtypes.int32)
        
        # --- CORRECT PAIN DELTA CALCULATION ---
        # Directly calculate the delta from the stored pain values.
        pain_deltas = [current - last for last, current in zip(last_pains, current_pains)]
        pain_deltas_tensor = Tensor(pain_deltas, device=Device.DEFAULT).realize()
        
        # --- Call the JIT function with the corrected data ---
        self._learn_from_experience_jit(last_perceptions_batch, current_perceptions_batch, actions_batch, memory_tensor, pain_deltas_tensor)
        
        # --- CORRECTLY RESET THE AGENT'S PAIN SIGNAL FOR THE NEXT DAY ---
        # After sleeping, the agent is idle. This sets the baseline for the next day's first action.
        current_tile_after_sleep = tuple(self.current_map[self.y, self.x])
        self.last_pain_signal = self.calculate_pain_signal(current_tile_after_sleep, 4, False) # Action 4 = idle, no wall hit

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
            num_rooms=400, trap_ratio=0.02, wind_ratio=0.15
        )
        self.current_map = self.static_map.copy()

        self.timestep = 0
        self.is_daytime = True
        
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

    def spawn_agent(self, parent: Agent = None):
        if not self.available_spawn_points: return

        spawn_pos = None
        
        # --- NEW: Proximity Spawning Logic ---
        if parent:
            search_radius = 5 # Search in an 11x11 square around the parent
            py, px = parent.y, parent.x
            
            nearby_spots = []
            for r in range(-search_radius, search_radius + 1):
                for c in range(-search_radius, search_radius + 1):
                    ny, nx = py + r, px + c
                    # Check if the spot is valid: within bounds, a floor, and not occupied
                    if (0 <= ny < self.height and 0 <= nx < self.width and
                        tuple(self.static_map[ny, nx]) == COLORS["FLOOR"] and
                        (ny, nx) not in self._agent_positions):
                        nearby_spots.append((ny, nx))
            
            if nearby_spots:
                spawn_pos = random.choice(nearby_spots)
                # print(f"Spawning child of {parent.id} at nearby location {spawn_pos}") # Optional: for debugging
        
        # --- FALLBACK: If no parent or no nearby spots, use original random logic ---
        if spawn_pos is None:
            spawn_pos = tuple(random.choice(self.available_spawn_points))
            while spawn_pos in self._agent_positions:
                spawn_pos = tuple(random.choice(self.available_spawn_points))

        child_weights = None
        if parent:
            parent_weights = get_state_dict(parent.model)
            child_weights = {}
            for name, tensor in parent_weights.items():
                noise_np = np.random.randn(*tensor.shape).astype(np.float32) * 0.01
                mutated_np = tensor.numpy() + noise_np
                child_weights[name] = Tensor(mutated_np, device="CPU").realize()
        
        initial_tile_color = tuple(self.static_map[spawn_pos])
        
        agent = Agent(self.next_agent_id, spawn_pos, initial_tile_color, self.config, initial_weights=child_weights)
        
        self.agents[self.next_agent_id] = agent
        
        self._agent_positions.add(spawn_pos)
        self.next_agent_id += 1

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
        # --- 1. Day/Night Cycle Management ---
        self.timestep += 1
        cycle_time = self.timestep % (DAY_LENGTH + NIGHT_LENGTH)
        
        if self.is_daytime and cycle_time >= DAY_LENGTH:
            self.is_daytime = False
            print(f"\n--- Timestep {self.timestep}: Night has fallen. Agents are sleeping and learning. ---")
        elif not self.is_daytime and cycle_time < DAY_LENGTH:
            self.is_daytime = True
            print(f"\n--- Timestep {self.timestep}: Day has broken. Agents are waking up. ---")

        agents_to_remove = []
        action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)} # up, down, left, right, idle
        WALL_BUMP_PENALTY = 5 
        
        # Pass the current map to agent methods that need it for state access
        for agent in self.agents.values():
            agent.current_map = self.current_map 

        if self.is_daytime:
            # --- DAYTIME: Agents act and collect experience ---
            perceptions = {
                agent_id: agent.get_perception(self.current_map) 
                for agent_id, agent in self.agents.items()
            }

            for agent_id, agent in self.agents.items():
                if not agent.is_alive: continue
                agent.is_sleeping = False
                agent.age += 1
                
                # --- NEW: Calculate pain BEFORE the action ---
                # The "last pain" is the pain of the state the agent is currently in,
                # resulting from its previous action.
                last_tile_color = tuple(self.current_map[agent.y, agent.x])
                # We assume the state we are in was not the result of a wall bump. This is a safe baseline.
                last_pain = agent.calculate_pain_signal(last_tile_color, agent.action_history[-1], False)

                # --- Existing Logic ---
                last_perception = perceptions[agent_id]
                history_for_experience = list(agent.action_history)
                action_idx = agent.choose_action(last_perception)
                
                dy, dx = action_map.get(action_idx, (0, 0))
                
                ny, nx = agent.y + dy, agent.x + dx
                hit_a_wall = False
                
                if not (0 <= ny < self.height and 0 <= nx < self.width and
                        not np.array_equal(self.current_map[ny, nx], COLORS["WALL"]) and
                        (ny, nx) not in self._agent_positions):
                    if action_idx != 4:
                        hit_a_wall = True
                        agent.energy -= WALL_BUMP_PENALTY
                    dy, dx = 0, 0 
                
                self._agent_positions.remove((agent.y, agent.x))
                agent.move(dy, dx)
                self._agent_positions.add((agent.y, agent.x))
                
                agent.action_history.append(action_idx)

                tile_color = tuple(self.current_map[agent.y, agent.x])
                
                # Check for death conditions
                if tile_color == COLORS["TRAP"]:
                    agent.is_alive = False
                    agents_to_remove.append(agent_id)
                    continue

                if tile_color == COLORS["WIND"]: agent.energy -= 10
                if tile_color == COLORS["POWER_CELL"]:
                    agent.energy += 25
                    self.current_map[agent.y, agent.x] = COLORS["FLOOR"]
                
                current_perception = agent.get_perception(self.current_map)

                agent.energy -= 1 

                # --- NEW: Calculate pain AFTER the action ---
                current_pain = agent.calculate_pain_signal(tile_color, action_idx, hit_a_wall)
                
                # Update the agent's live pain signal for metric tracking
                agent.last_pain_signal = current_pain

                # --- Store the new, correct experience tuple ---
                current_perception = agent.get_perception(self.current_map)
                agent.experience_buffer.append(
                    (last_perception, action_idx, current_perception, last_pain, current_pain, history_for_experience)
                )
                
                if agent.energy <= 0:
                    agent.is_alive = False
                    agents_to_remove.append(agent_id)

        else:
            # --- NIGHTTIME: Agents sleep and learn ---
            for agent_id, agent in self.agents.items():
                if not agent.is_alive: continue
                agent.is_sleeping = True
                
                # Learn from a batch of memories
                agent.learn_from_experience(batch_size=BATCH_SIZE)
                
                # Small energy cost to survive the night
                agent.energy -= 0.25 

                if agent.energy <= 0:
                    agent.is_alive = False
                    agents_to_remove.append(agent_id)

        gc.collect()
                
        # --- Agent Respawn Logic (remains the same) ---
        if agents_to_remove:
            surviving_agents = [a for a in self.agents.values() if a.is_alive]
            parent = None
            if surviving_agents:
                tournament_size = min(3, len(surviving_agents))
                contestants = random.sample(surviving_agents, tournament_size)
                parent = max(contestants, key=lambda a: a.age)

            for agent_id in agents_to_remove:
                dead_agent = self.agents[agent_id]
                if dead_agent.age > self.oldest_agent_ever_age:
                    self.oldest_agent_ever_age = dead_agent.age
                    self.oldest_agent_ever = dead_agent.id
                
                pos = (dead_agent.y, dead_agent.x)
                if pos in self._agent_positions: self._agent_positions.remove(pos)
                
                # Respawn a new agent, passing the winning parent object
                self.spawn_agent(parent=parent)
                del self.agents[agent_id]

            gc.collect()       
        
        if self.is_daytime and random.random() < 0.05:
            self.spawn_power_cells()

        gc.collect()

    def get_population_metrics(self):
        """Calculates and returns key metrics about the agent population."""
        live_agents = [a for a in self.agents.values() if a.is_alive]
        if not live_agents:
            return {
                "avg_age": 0, "max_live_age": 0, "oldest_ever": self.oldest_agent_ever,
                "age_variance": 0, "avg_pain": 0 # NEW: Default values
            }
            
        ages = [a.age for a in live_agents]
        pains = [a.last_pain_signal for a in live_agents] # NEW

        avg_age = sum(ages) / len(ages)
        avg_pain = sum(pains) / len(pains) # NEW
        age_variance = np.var(ages) if len(ages) > 1 else 0 # NEW
        
        oldest_living_agent = max(live_agents, key=lambda a: a.age)
        max_live_age = oldest_living_agent.age
        
        return {
            "avg_age": avg_age,
            "max_live_age": max_live_age,
            "oldest_ever_id": self.oldest_agent_ever,
            "oldest_ever_age": self.oldest_agent_ever_age,
            "age_variance": age_variance, # NEW
            "avg_pain": avg_pain # NEW
        }

    def get_render_frame(self, pixel_per_tile=50):
        """Creates a render frame with a specified pixel size per tile."""
        frame = np.kron(self.current_map, np.ones((pixel_per_tile, pixel_per_tile, 1), dtype=np.uint8))

        for agent in self.agents.values():
            if not agent.is_alive: continue
            
            y_start, x_start = agent.y * pixel_per_tile, agent.x * pixel_per_tile
            frame[y_start:y_start+pixel_per_tile, x_start:x_start+pixel_per_tile] = agent.color
        
        return frame
    
def render_episode(world: World, episode_num: int, num_steps: int, timestamps: List = [], avg_ages: List = [], max_live_ages: List = [], oldest_ever_ages: List = [], age_variances: List = [], avg_pains: List = []):
    """
    Runs the simulation and renders output, now with more detailed plots.
    """
    print(f"\n--- Starting Episode {episode_num} ---")
    
    PIXEL_PER_TILE = 50
    
    fig = plt.figure(figsize=(24, 12))
    # GridSpec: 2 rows, 3 columns. Maze takes up all rows in the first column.
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], wspace=0.3, hspace=0.4)
    
    ax_maze = fig.add_subplot(gs[:, 0])
    ax_age = fig.add_subplot(gs[0, 1])
    ax_pain = fig.add_subplot(gs[0, 2])
    ax_variance = fig.add_subplot(gs[1, 1])
    ax_time = fig.add_subplot(gs[1, 2]) # A plot for time per step if desired

    im = ax_maze.imshow(world.get_render_frame(pixel_per_tile=PIXEL_PER_TILE), animated=True)
    ax_maze.set_xticks([])
    ax_maze.set_yticks([])
    agent_texts = {}

    # Plot 1: Age Metrics
    ax_age.set_title("Population Age")
    ax_age.set_xlabel("Timestep")
    ax_age.set_ylabel("Age")
    line_avg, = ax_age.plot([], [], label='Avg Age', color='cyan')
    line_max, = ax_age.plot([], [], label='Oldest Living', color='lime')
    line_ever, = ax_age.plot([], [], label='Oldest Ever', color='magenta', linestyle='--')
    ax_age.legend()
    ax_age.grid(True, alpha=0.3)

    # Plot 2: Pain Metric
    ax_pain.set_title("Average Population Pain")
    ax_pain.set_xlabel("Timestep")
    ax_pain.set_ylabel("Pain Signal")
    line_pain, = ax_pain.plot([], [], label='Avg Pain', color='red')
    ax_pain.legend()
    ax_pain.grid(True, alpha=0.3)

    # Plot 3: Variance Metric
    ax_variance.set_title("Population Age Variance")
    ax_variance.set_xlabel("Timestep")
    ax_variance.set_ylabel("Variance")
    line_var, = ax_variance.plot([], [], label='Age Variance', color='yellow')
    ax_variance.legend()
    ax_variance.grid(True, alpha=0.3)

    # (Optional) You can use the 4th plot for time_per_step or other metrics
    ax_time.set_title("Processing Time")
    ax_time.set_xlabel("Timestep")
    ax_time.set_ylabel("Seconds per Step")
    ax_time.grid(True, alpha=0.3)

    def update(frame_num):
        start_time = time.time()
        world.step()
        end_time = time.time()
        time_per_step = end_time - start_time
        
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
        global_timestep = world.timestep # Use world's persistent timestep
        
        # Log all metrics to wandb
        wandb.log({
            "timestep": global_timestep,
            "time_per_step_s": time_per_step,
            "avg_population_age": metrics['avg_age'],
            "oldest_living_age": metrics['max_live_age'],
            "oldest_ever_age": metrics['oldest_ever_age'],
            "age_variance": metrics['age_variance'], # NEW LOG
            "avg_pain": metrics['avg_pain']          # NEW LOG
        })

        # Update data for all plots
        timestamps.append(global_timestep)
        avg_ages.append(metrics['avg_age'])
        max_live_ages.append(metrics['max_live_age'])
        oldest_ever_ages.append(metrics['oldest_ever_age'])
        age_variances.append(metrics['age_variance']) # NEW
        avg_pains.append(metrics['avg_pain'])         # NEW

        # Update Age Plot
        line_avg.set_data(timestamps, avg_ages)
        line_max.set_data(timestamps, max_live_ages)
        line_ever.set_data(timestamps, oldest_ever_ages)
        
        # Update New Plots
        line_pain.set_data(timestamps, avg_pains)
        line_var.set_data(timestamps, age_variances)

        # Rescale all plot axes
        for ax in [ax_age, ax_pain, ax_variance]:
            ax.relim()
            ax.autoscale_view()

        # Print progress to console
        if (frame_num + 1) % 100 == 0:
            print(f"  Episode {episode_num}, Step {frame_num+1}/{num_steps} | "
                  f"Oldest Ever: {metrics['oldest_ever_age']}")

        # Return all animated artists
        return [im] + list(agent_texts.values()) + [line_avg, line_max, line_ever, line_pain, line_var]

    # --- Create and Save the Animation ---
    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=150, blit=True, repeat=False)
    
    output_filename = f"episode_{episode_num}.gif"
    ani.save(output_filename, writer='pillow', fps=10)
    plt.close(fig)

    # Return the updated lists
    return timestamps, avg_ages, max_live_ages, oldest_ever_ages, age_variances, avg_pains

# --- Main Simulation and Animation Setup ---
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    GRID_WIDTH = 101 
    GRID_HEIGHT = 101
    NUM_AGENTS = 64
    NUM_POWER_CELLS = 256

    DAY_LENGTH = 16
    NIGHT_LENGTH = 4
    BATCH_SIZE = 8 # Number of memories to learn from each night

    world = World(width=GRID_WIDTH, height=GRID_HEIGHT, num_agents=NUM_AGENTS, num_power_cells=NUM_POWER_CELLS)

    TOTAL_SIMULATION_STEPS = 10000
    STEPS_PER_EPISODE = 10 # This will create animations of 100 steps each
    
    num_episodes = TOTAL_SIMULATION_STEPS // STEPS_PER_EPISODE

    ## WANDB CHANGE: 2. Initialize wandb run
    wandb.init(
        project="pain-signal-hypothesis",
        config={
            "grid_width": GRID_WIDTH,
            "grid_height": GRID_HEIGHT,
            "num_agents": NUM_AGENTS,
            "num_power_cells": NUM_POWER_CELLS,
            "total_simulation_steps": TOTAL_SIMULATION_STEPS,
            "steps_per_episode": STEPS_PER_EPISODE,
        }
    )

    timestamps, avg_ages, max_live_ages, oldest_ever_ages, age_variances, avg_pains = [], [], [], [], [], []

    for i in range(num_episodes):
        # Pass and receive the new lists
        timestamps, avg_ages, max_live_ages, oldest_ever_ages, age_variances, avg_pains = render_episode(
            world, episode_num=i + 1, num_steps=STEPS_PER_EPISODE, 
            timestamps=timestamps, avg_ages=avg_ages, max_live_ages=max_live_ages, 
            oldest_ever_ages=oldest_ever_ages, age_variances=age_variances, avg_pains=avg_pains
        )

    ## WANDB CHANGE: 3. Finish the wandb run
    wandb.finish()
    print("\n--- All episodes rendered. Wandb run finished. ---")