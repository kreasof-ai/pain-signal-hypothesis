# **Experiment: The Homeostatic Imperative in Embodied Intelligence - A Pain Signal Hypothesis**

**Objective:**

To demonstrate that an embodied agent driven by a pain signal, composed of uncertainty, computational load, and energy level, can learn to survive and adapt in a complex environment without explicit external rewards.

**Hypothesis:**

An agent that seeks to minimize its pain signal will exhibit emergent behaviors such as exploration, efficient computation, and resource management, leading to long-term survival in a dynamic environment.

**Algorithm:**

**I. Environment Setup:**

![grid-world](Agent Simulation.mp4)

1. **Grid World:**
    *   Define grid size: 17 x 17.
    *   Define obstacle types:
        *   Walls (W): Impassable.
        *   Holes (H): Agent dies if it enters a hole.
        *   Wind (Y): Increases energy cost for movement and staying.
    *   Define power cell locations (P1-P14) and their appearance/disappearance schedules. Each power cell has a unique schedule, represented as a list of (start\_time, end\_time) tuples, indicating when it's available during a cycle (e.g., a 100-step cycle).
    *   Define the agent's starting/respawn position (S).

2. **Power Cell Dynamics:**
    *   Each power cell has a timer that determines when it disappears.
    *   When a power cell is collected, it provides a fixed amount of energy (e.g., 50) but does not allow the agent to exceed a maximum energy level (e.g., 200).
    *   Implement a mechanism for a random power cell to spawn within the agent's perception range when the agent's energy level falls below a threshold (e.g., 10) with a certain probability.

**II. Agent Setup:**

1. **Architecture:**
    *   **ResNet:** Processes the `K_grid x K_grid` local view centered on the agent.
        *   Input: `K_grid x K_grid` grid representation with one-hot encoding for each cell type (empty, wall, hole, wind, power cell).
        *   Output: Feature map representing the local environment.
    *   **Transformer:** Processes the sequence of the agent's previous actions.
        *   Input: Sequence of the last `N_steps` actions (represented as embeddings).
        *   Output: Action probabilities.
    *   **Cross-Attention:** Integrates the ResNet feature map with the Transformer's output to inform action selection.

2. **Parameters:**
    *   `K_grid`: Initial perception size (e.g., 3).
    *   `N_steps`: Initial memory length (e.g., 5).
    *   `w_u`: Weight for uncertainty in the pain signal.
    *   `w_c`: Weight for computational load in the pain signal.
    *   `w_e`: Weight for energy level in the pain signal.
    *   Learning rate, discount factor, exploration parameters (e.g., epsilon for epsilon-greedy).

3. **Internal State:**
    *   **Uncertainty:** Initialized to a high value (e.g., 1.0). Updated based on prediction errors.
    *   **Computational Load:** Initialized to 0. Increases with `K_grid` size, `N_steps`, and potentially the complexity of internal computations.
    *   **Energy Level:** Initialized to 100. Decreases with movement, perception, memory, and wind. Increases when a power cell is collected.

**III. Training Process:**

1. **Initialization:**
    *   Initialize the environment (grid, obstacles, power cell schedules).
    *   Initialize the agent (ResNet, Transformer, internal state).
    *   Initialize `survival_time` to 0.

2. **Main Loop (Run Indefinitely):**
    *   **a. Perception:**
        *   Agent observes the `K_grid x K_grid` area around it using the ResNet.
        *   Agent predicts the next `K_grid x K_grid` perception using the ResNet and the last hidden state of the Transformer.
    *   **b. Action Selection:**
        *   The Transformer processes the action history and the ResNet feature map (through cross-attention) to produce action probabilities.
        *   The agent selects an action based on these probabilities using an exploration strategy (e.g., epsilon-greedy).
        *   The agent may choose to change its perception size (`K_grid`) or memory length (`N_steps`).
    *   **c. Environment Interaction:**
        *   The agent executes the chosen action in the environment.
        *   The environment transitions to a new state based on the action and the grid dynamics (wind, power cell appearance/disappearance).
    *   **d. Reward and Done:**
        *   The agent receives a reward for collecting a power cell (e.g., +5).
        *   The agent receives a penalty for falling into a hole (e.g., -10).
        *   `done` is set to `True` if the agent falls into a hole or its energy reaches 0. Otherwise, `done` is `False`.
    *   **e. Prediction Error Calculation:**
        *   The agent compares its predicted next perception with the actual next perception.
        *   The prediction error is calculated (e.g., as the number of mismatched cells).
    *   **f. Internal State Update:**
        *   **Uncertainty:** Updated based on the prediction error.
        *   **Computational Load:** Updated based on `K_grid`, `N_steps`, and potentially other factors.
        *   **Energy Level:** Updated based on the action taken, wind effects, and whether a power cell was collected.
        *   **Action History:** The current action is added to the action history.
    *   **g. Pain Signal Calculation:**
        *   The pain signal is calculated as a weighted sum of uncertainty, computational load, and the inverse of the energy level: `P = w_u * U + w_c * C + w_e * (100 - E)`
    *   **h. Learning:**
        *   The ResNet is trained to predict the next perception using supervised learning (minimizing the prediction error).
        *   The Transformer is trained using reinforcement learning (e.g., actor-critic or Q-learning) to minimize the long-term pain signal.
    *   **i. Survival Tracking:**
        *   If the agent is alive, increment `survival_time`.
        *   If the agent dies, reset `survival_time` to 0 and respawn the agent at the starting position, keeping the learned parameters but resetting position and energy level.
    *   **j. Termination Check:**
        *   If `survival_time` is greater than or equal to 10,000, terminate the simulation.
    *   **k. Visualization (Optional):**
        *   Update real-time visualizations (Pygame, Matplotlib) or log data for later analysis.

**IV. Evaluation:**

1. **Metrics:**
    *   **Survival Time:** The primary metric - how many timesteps the agent survives consecutively.
    *   **Average Pain Signal:** Track the average pain signal over time.
    *   **Uncertainty, Computational Load, and Energy over Time:** Analyze the evolution of these components.
    *   **Number of Deaths:** How many times does the agent die before reaching the survival goal?
    *   **Power Cells Collected:** How many power cells does the agent collect on average?
    *   **Exploration Rate:** How much of the environment does the agent explore?
    *   **Average `K_grid` and `N_steps`:** How do these values change over time?

2. **Analysis:**
    *   Analyze the agent's behavior to see if it exhibits the expected emergent behaviors (exploration, risk aversion, pathfinding, computational trade-offs).
    *   Investigate the relationship between the pain signal components and the agent's performance.
    *   Compare the performance of agents with different pain signal weights (`w_u`, `w_c`, `w_e`).
    *   Compare the performance of agents with different architectures or learning algorithms.

**V. Visualization:**

1. **Real-time (using Pygame):**
    *   Display the grid world.
    *   Show the agent's position and movement.
    *   Visualize the agent's perception area (`K_grid x K_grid`).
    *   Indicate the locations of walls, holes, wind, and power cells.
    *   Display the agent's current energy level, and potentially the uncertainty and computational load.

2. **Post-Simulation (using Matplotlib or TensorBoard/Visdom):**
    *   Plot the pain signal and its components over time.
    *   Plot the agent's survival time over multiple episodes.
    *   Create heatmaps of the agent's uncertainty about the environment.
    *   Visualize the agent's learned policy (e.g., using arrows to indicate the preferred action in each cell).
    *   Visualize the weights or activations of the ResNet and Transformer to gain insights into what the agent has learned.
