from torch import optim
from collections import deque
import torch
from torch import nn
from network import Network
import numpy as np
import random


class DQNAgentBase:
    # Epsilon decay so that we reach min value in 500 episodes
    def __init__(self, env, trainable: bool, domain_reward_factor=1.0, gamma=0.99, epsilon=0.9,
                 epsilon_decay=0.99997, epsilon_min=0.05, batch_size=64, learning_rate=0.001, sync_rate=1000,
                 optimizer=optim.AdamW):

        # Initialize agent
        self.trainable = trainable
        self.domain_reward_factor = domain_reward_factor
        self.action_space = env.action_space
        self.replay_buffer = deque([], maxlen=10000)  # replay buffer with fixed size of 10000
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay  # 0.99 -> reach minimum after >~ 300 episodes
        self.epsilon_min = epsilon_min  # minimum value of epsilon
        self.gamma = gamma  # discount factor
        self.batch_size = batch_size  # batch size for training (< truncation !!)
        self.sync_rate = sync_rate  # rate at which to sync target network with policy network
        self.loss = nn.MSELoss()  # loss function

        # Initialize Q-networks separately otherwise we update both networks
        self.policy_net = Network(input_states=env.observation_space.shape[0], out_actions=self.action_space.n,
                                  hidden_layers=2, hidden_nodes=64)
        self.target_net = Network(input_states=env.observation_space.shape[0], out_actions=self.action_space.n,
                                  hidden_layers=2, hidden_nodes=64)

        if not trainable:
            # Load the best network
            self.policy_net.load_state_dict(torch.load(f'trained_models/{self.__class__.__name__}.pth'))

        self.optimizer = optimizer(self.policy_net.parameters(), lr=learning_rate)

        # Copy the two networks weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Track training stats
        self.steps = 0

        self.curr_episode_length = 0

        self.curr_episode_dqn_loss = 0.0
        self.episodes_dqn_losses = []

        self.curr_episode_env_reward = 0.0
        self.episodes_env_rewards = []

    def observe(self, state, action, next_state, reward, terminated, truncated):
        # Increment the agent's steps count
        self.steps += 1
        self.curr_episode_length += 1

        # Store the reward
        self.curr_episode_env_reward += reward

        # Update agent's internal memory with observed transition
        self.replay_buffer.append((state, action, next_state, reward, terminated))

        # Update the agent's networks
        if self.trainable:
            self.update()

        # End of episode
        if terminated or truncated:
            self.episodes_dqn_losses.append(self.curr_episode_dqn_loss / self.curr_episode_length)
            self.episodes_env_rewards.append(self.curr_episode_env_reward)

            self.curr_episode_length = 0
            self.curr_episode_dqn_loss = 0
            self.curr_episode_env_reward = 0

    def select_action(self, state):
        # Select action based on epsilon-greedy policy

        # Exponentially decaying epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():  # no need to calculate gradients
                state_tensor = torch.tensor(state)
                action = torch.argmax(self.policy_net(state_tensor)).item()
        return action

    def update(self):
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
            # Add the step loss to the episode's loss
            self.curr_episode_dqn_loss += loss.item()
            # Update policy network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # synchronize the policy and target network
            if self.steps > 0 and self.steps % self.sync_rate == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self):
        if self.trainable:
            torch.save(self.policy_net.state_dict(), f'trained_models/{self.__class__.__name__}.pth')

    def get_training_dict(self):
        return {
            "episodes_mean_dqn_loss": self.episodes_dqn_losses,
            "episodes_environment_rewards": self.episodes_env_rewards,
            "steps_dqn_loss": self.episodes_dqn_losses
        }
