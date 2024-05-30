import numpy as np


class DynaAgent:
    def __init__(self, env, trainable: bool, k=10, discr_step=(0.025, 0.005), gamma=0.99, epsilon=0.9,
                 epsilon_decay=0.99999, epsilon_min=0.05, snapshot_episodes=(100, 500, 1000, 3000)):
        # Initialize agent
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.trainable = trainable
        self.k = k  # number of updates
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay  # decaying rate of epsilon
        self.epsilon_min = epsilon_min  # minimum value of epsilon
        self.snapshot_episodes = snapshot_episodes

        # Create the bins based on the discr_step
        self.bins = [np.arange(self.observation_space.low[i], self.observation_space.high[i], discr_step[i]) for i in
                     range(self.observation_space.shape[0])]
        n_states = (len(self.bins[0]) - 1) * (len(self.bins[1]) - 1)
        # Fix the bins limits because they are not precise

        # Create the arrays
        # 1. Arrays to compute P_hat and R_hat
        self.state_action_count = np.zeros((n_states, self.action_space.n))  # count of state-action pairs
        self.transition_count = np.zeros((n_states, self.action_space.n, n_states))  # count of transitions (s, a) -> s'
        # 2. Arrays to compute Q-values and choose actions
        self.P_hat = np.ones((n_states, self.action_space.n, n_states)) * (1 / n_states)  # estimated transition probs.
        self.R_hat = np.zeros((n_states, self.action_space.n))  # expected rewards (initialisation does not matter ?)
        self.Q = np.zeros((n_states, self.action_space.n))  # estimated Q-values
        if not trainable:
            self.Q = np.load(f'trained_models/{self.__class__.__name__}.npy')

        # Keep track of the last choice the agent made
        self.current_transition = None

        # Track training stats
        self.episodes = 0

        self.steps_q_values_updates = []
        self.q_values_snapshots = []

        self.curr_episode_trajectory = []
        self.trajectories_snapshots = []

        self.curr_episode_length = 0
        self.episodes_lengths = []

        self.curr_episode_env_reward = 0.0
        self.episodes_env_rewards = []

        self.success_count = 0
        self.cumulative_successes = []

    def discretize_state(self, state):
        # Return the unique number and the 2d-state associated to the state after discretization
        dis_state_2d = [np.digitize(state[i], self.bins[i]) - 1 for i in range(self.observation_space.shape[0])]
        dis_state_id = dis_state_2d[0] * (len(self.bins[1]) - 1) + dis_state_2d[1]
        return dis_state_id, dis_state_2d

    def observe(self, state, action, next_state, reward, terminated, truncated):
        # Increment the agent's steps count
        self.curr_episode_length += 1

        # Update the successes count if terminated
        if terminated:
            self.success_count += 1

        # Store the reward
        self.curr_episode_env_reward += reward

        # Discretize the states
        dis_state_id, dis_state_2d = self.discretize_state(state)
        dis_next_state_id, _ = self.discretize_state(next_state)

        # Increment counters
        self.state_action_count[dis_state_id, action] += 1
        self.transition_count[dis_state_id, action, dis_next_state_id] += 1

        # Store the current state-action pair and reward, used in update()
        self.current_transition = (dis_state_id, action, reward)

        # Store the trajectory
        self.curr_episode_trajectory.append(dis_state_2d)

        # Update the agent's Q-values
        if self.trainable:
            self.update()

        # End of episode
        if terminated or truncated:
            self.episodes_lengths.append(self.curr_episode_length)
            self.episodes_env_rewards.append(self.curr_episode_env_reward)
            self.cumulative_successes.append(self.success_count)

            self.episodes += 1

            # Take snapshots of Q-values and trajectories
            if self.episodes in self.snapshot_episodes:
                # Put the Q-values in a 2D array
                q_values = np.reshape(np.max(self.Q, axis=1), ((len(self.bins[0])) - 1, -1))
                # Keep only the values of visited states
                q_values[(self.state_action_count.sum(axis=1) == 0).reshape((len(self.bins[0])) - 1, -1)] = np.nan
                # Save the Q-values snapshot
                self.q_values_snapshots.append(q_values)
                # Save the trajectory
                self.trajectories_snapshots.append(self.curr_episode_trajectory.copy())

            self.curr_episode_length = 0
            self.curr_episode_env_reward = 0
            self.curr_episode_trajectory = []

    def select_action(self, state):
        # Select action based on epsilon-greedy policy
        dis_state_id, _ = self.discretize_state(state)

        # Exponentially decaying epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Choose action randomly or based on Q-values
        if np.random.rand() < self.epsilon and self.trainable:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Q[dis_state_id, :])

        return action

    def update(self):
        if self.trainable:

            # Unpack current transition
            (dis_state, action, reward) = self.current_transition

            # Update the transition probabilities
            self.P_hat[dis_state, action, :] = self.transition_count[dis_state, action, :] / self.state_action_count[
                dis_state, action]

            # Update the expected reward using a running average
            self.R_hat[dis_state, action] = ((self.state_action_count[dis_state, action] - 1) * self.R_hat[
                dis_state, action] + reward) / self.state_action_count[dis_state, action]

            # Update the Q-value
            old_q_val = self.Q[dis_state, action]
            self.Q[dis_state, action] = self.R_hat[dis_state, action] + self.gamma * np.sum(
                self.P_hat[dis_state, action, :] * np.max(self.Q, axis=1))

            # Keep track of the Q-values updates
            update_sum = np.abs(old_q_val - self.Q[dis_state, action])

            # Perform additional updates on already encountered state-action pairs
            # Get the already encountered pairs
            encountered_states_actions = np.argwhere(self.state_action_count > 0)

            # Choose k random pairs among them
            random_idx = np.random.randint(low=0, high=len(encountered_states_actions), size=self.k)
            random_states_actions = encountered_states_actions[random_idx]

            # Update the Q-value for those pairs
            for state_action in random_states_actions:
                dis_state = state_action[0]
                action = state_action[1]
                old_q_val = self.Q[dis_state, action]
                self.Q[dis_state, action] = self.R_hat[dis_state, action] + self.gamma * np.sum(
                    self.P_hat[dis_state, action, :] * np.max(self.Q, axis=1))
                update_sum += np.abs(old_q_val - self.Q[dis_state, action])

            # Store the average update of Q-values for this step
            self.steps_q_values_updates.append(update_sum / (self.k + 1))

    def save(self):
        if self.trainable:
            np.save(f'trained_models/{self.__class__.__name__}.npy', self.Q)

    def get_training_dict(self):
        return {
            "episodes_lengths": self.episodes_lengths,
            "episodes_environment_rewards": self.episodes_env_rewards,
            "cumulative_successes": self.cumulative_successes,
            # Potentially change this, but should work with plots functions
            "steps_dqn_loss": self.steps_q_values_updates,
            "q_values_snapshots": self.q_values_snapshots,
            "trajectories_snapshots": self.trajectories_snapshots
        }
