# Rolls-Royce-Peak-Plower
Deep Q-learning (DQN) and Dyna implementations to train the Gymnasium 'MountainCar-v0' environment with continuous 2D phase-space.

Practical comparison of a model-free and a model-based approach to RL with the Mountaincar environment.

Implementations of:
- basic DQN Agent
- DQN agent with heuristic reward based on environment knowledge
- DQN agent with Random Network Distillation (RND)
- Dyna-Q agent

# Quick summary:

<u>DQN:</u>
- Basic implementation consists of a target-, and policy deep neural network, that use replay experience to update according to the TD-error from Q-learning. The target is fixed for a specified sync rate to suppress noise from online-updating. A $\varepsilon$-greedy policy is chosen with exponential $\varepsilon$ decay.
- An auxiliary reward based on total energy is introduced, to incentivize the agent to reach the goal. Upper bound for reward is set at $r_A=1$ to prevent positive reward states.
- A RND based agent is introduced, based on [Burda et al., 2018](https://arxiv.org/pdf/1810.12894). Two additional NN's are added and randomly initialised to add an intrinsic reward based on frequency of state visitation. 

<u>Dyna-Q:</u>
- Dyna-Q models the environment by estimating transition probabilities. Implementation is based on a Q-table, the phase-space has to be finite and therefore quantisation is required. The same $\varepsilon$-greedy policy as in DQN is used.

