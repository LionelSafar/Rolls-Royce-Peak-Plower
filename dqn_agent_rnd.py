from torch import optim
from collections import deque
import torch
from torch import nn
from network import Network
import numpy as np
import random


class DQNAgentRND:
    def __init__(self, env, trainable: bool, reward_factor=1.0, gamma=0.99, epsilon=0.9, epsilon_decay=0.99999,
                 epsilon_min=0.05, batch_size=64, learning_rate=0.001, sync_rate=1000, optimizer=optim.AdamW):

        # Initialize agent
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.trainable = trainable
        self.reward_factor = reward_factor
        self.replay_buffer = deque([], maxlen=10000)  # replay buffer with fixed size of 10000
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay  # decaying rate of epsilon
        self.epsilon_min = epsilon_min  # minimum value of epsilon
        self.gamma = gamma  # discount factor
        self.batch_size = batch_size  # batch size for training
        self.sync_rate = sync_rate  # rate at which to sync target network with policy network
        self.loss = nn.MSELoss()  # loss function
        self.steps = 0

        # Initialize Q-networks separately otherwise we update both networks
        self.policy_net = Network(input_states=self.observation_space.shape[0], out_actions=self.action_space.n,
                                  hidden_layers=2, hidden_nodes=64)
        self.target_net = Network(input_states=self.observation_space.shape[0], out_actions=self.action_space.n,
                                  hidden_layers=2, hidden_nodes=64)

        # Additionally initialise predictor and target predictor networks
        self.predictor_net = Network(input_states=self.observation_space.shape[0], out_actions=1,
                                     hidden_layers=2, hidden_nodes=64)
        self.target_predictor_net = Network(input_states=self.observation_space.shape[0], out_actions=1,
                                            hidden_layers=2, hidden_nodes=64)

        if not trainable:  # load the best network
            self.policy_net.load_state_dict(torch.load(f'trained_models/{self.__class__.__name__}.pth'))
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_optimizer = optimizer(self.policy_net.parameters(), lr=learning_rate)
        self.predictor_optimizer = optimizer(self.predictor_net.parameters(), lr=learning_rate)

        # Running estimates of rewards and states
        self.states_mean = torch.tensor([0.0] * self.observation_space.shape[0])
        self.states_M2 = torch.tensor(
            [0.0] * self.observation_space.shape[0])  # sum of squared distances from the mean (to compute running std)
        self.rnd_loss_mean = 0.0
        self.rnd_loss_M2 = 0.0  # sum of squared distances from the mean (to compute running std)

        # Keep track of the loss (computed in observe() but backward prop. in update())
        self.rnd_loss = None

        # Track training stats
        self.steps = 0
        self.steps_dqn_losses = []

        self.curr_episode_length = 0
        self.episodes_lengths = []

        self.curr_episode_env_reward = 0.0
        self.episodes_env_rewards = []

        self.curr_episode_aux_reward = 0.0
        self.episodes_aux_rewards = []

        self.curr_episode_reward = 0.0
        self.episodes_rewards = []

        self.success_count = 0
        self.cumulative_successes = []

    def observe(self, state, action, next_state, reward, terminated, truncated):
        # Increment the agent's steps count
        self.steps += 1
        self.curr_episode_length += 1

        # Update the successes count if terminated
        if terminated:
            self.success_count += 1

        # Compute the intrinsic reward, store the loss for backward prop. in update()
        rnd_reward, self.rnd_loss = self.compute_rnd_reward(next_state)

        # Update the reward with factor (reward is used only after a few steps)
        rnd_reward = rnd_reward * self.reward_factor if len(self.replay_buffer) > self.batch_size else 0.0

        # Store the rewards
        self.curr_episode_env_reward += reward
        self.curr_episode_aux_reward += rnd_reward
        reward += rnd_reward
        self.curr_episode_reward += reward

        # Update agent's internal memory with observed transition
        self.replay_buffer.append((state, action, next_state, reward, terminated))

        # Update the agent's networks
        if self.trainable:
            self.update()

        # End of episode
        if terminated or truncated:
            self.episodes_lengths.append(self.curr_episode_length)
            self.episodes_env_rewards.append(self.curr_episode_env_reward)
            self.episodes_aux_rewards.append(self.curr_episode_aux_reward)
            self.episodes_rewards.append(self.curr_episode_reward)
            self.cumulative_successes.append(self.success_count)

            self.curr_episode_length = 0
            self.curr_episode_env_reward = 0
            self.curr_episode_aux_reward = 0
            self.curr_episode_reward = 0

    def select_action(self, state):
        # Select action based on epsilon-greedy policy -> returns action

        # Exponentially decaying epsilon
        exploration_prob = max(self.epsilon * np.exp(-self.steps / self.epsilon_decay), self.epsilon_min)

        if np.random.rand() < exploration_prob:
            action = self.action_space.sample()
        else:
            with torch.no_grad():  # no need to calculate gradients
                state_tensor = torch.tensor(state)
                action = torch.argmax(
                    self.policy_net(state_tensor)).item()  # exploit with probability 1-epsilon from policy network
        return action

    def update(self):
        # Update the agent

        # RND (Backward)
        # Update the predictor network
        # We do not do it for the replay buffer: https://arxiv.org/pdf/1905.07579 section IV C.
        self.predictor_optimizer.zero_grad()
        self.rnd_loss.backward()
        self.predictor_optimizer.step()

        # DQN
        if len(self.replay_buffer) > self.batch_size:
            # Sample mini-batch from memory, don't use for loop
            mini_batch = random.sample(self.replay_buffer, self.batch_size)

            # Unpack mini-batch
            states = torch.tensor([transition[0] for transition in mini_batch], dtype=torch.float)
            actions = torch.tensor([transition[1] for transition in mini_batch], dtype=torch.long)
            next_states = torch.tensor([transition[2] for transition in mini_batch], dtype=torch.float)
            rewards = torch.tensor([transition[3] for transition in mini_batch], dtype=torch.float)
            terminated = torch.tensor([transition[4] for transition in mini_batch], dtype=torch.bool)

            with torch.no_grad():
                # Calculate target values
                Q_target = torch.zeros(rewards.shape)
                # Solve the issue of terminal states -> target = reward if terminated as Q-value is 0
                Q_target[terminated] = rewards[terminated]
                # Note that self.target_net(next_states_tensor[~terminated]).shape = (batch_size, action_space)
                Q_target[~terminated] = rewards[~terminated] + self.gamma * torch.max(
                    self.target_net(next_states[~terminated]), dim=1).values

            # Calculate the current Q-values
            Q_current = self.policy_net(states)[range(self.batch_size), actions]

            # Calculate loss
            loss = self.loss(Q_current, Q_target)
            # Store the step loss
            self.steps_dqn_losses.append(loss.item())
            # Update policy network
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

            # synchronize the policy and target network
            if self.steps > 0 and self.steps % self.sync_rate == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def compute_rnd_reward(self, state):
        # Helper function used in observe() to compute the RND reward given a state.
        # Does not do backward pass (done in update()), only forward pass.

        # Update the states running mean and std (needs to be done from the start)
        state = torch.tensor(state, dtype=torch.float)
        self.states_mean, self.states_M2 = self.compute_running_avg_and_M2(state, self.states_mean, self.states_M2)

        # Normalize the state
        states_std = np.sqrt(self.states_M2 / self.steps)
        normalized_state = (
            (state - self.states_mean) / states_std if all(el != 0 for el in states_std) else torch.zeros_like(state)
        )

        # Run the state through predictor and target networks
        predictor_out = self.predictor_net(normalized_state)
        target_out = self.target_predictor_net(normalized_state)

        # Get the mean square loss between both networks' outputs
        rnd_loss = self.loss(predictor_out, target_out)
        rnd_loss_val = rnd_loss.item()

        # Update the RND loss running mean and std (needs to be done from the start)
        self.rnd_loss_mean, self.rnd_loss_M2 = self.compute_running_avg_and_M2(rnd_loss_val, self.rnd_loss_mean,
                                                                               self.rnd_loss_M2)

        # Standardize and clamp the loss to get reward
        rnd_loss_std = np.sqrt(self.rnd_loss_M2 / self.steps)
        reward = (rnd_loss_val - self.rnd_loss_mean) / rnd_loss_std if rnd_loss_std != 0 else 0.0
        reward = np.clip(reward, -5, 5)

        return reward, rnd_loss

    def compute_running_avg_and_M2(self, new_val, current_avg, current_M2):
        # Helper function used in update() to compute the new running average
        # and update the sum of squares of differences from the current mean.
        # Based on https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        # append the new value to the mean
        new_avg = (self.steps - 1) / self.steps * current_avg + new_val / self.steps

        # Compute the new sum of squares of differences from the current mean
        new_M2 = current_M2 + (new_val - current_avg) * (new_val - new_avg)

        return new_avg, new_M2

    def save(self):
        if self.trainable:
            torch.save(self.policy_net.state_dict(),
                       f'trained_models/{self.__class__.__name__}.pth')

    def get_training_dict(self):
        return {
            "episodes_lengths": self.episodes_lengths,
            "episodes_environment_rewards": self.episodes_env_rewards,
            "episodes_auxiliary_rewards": self.episodes_aux_rewards,
            "episodes_rewards": self.episodes_rewards,
            "cumulative_successes": self.cumulative_successes,
            "steps_dqn_loss": self.steps_dqn_losses,
        }
