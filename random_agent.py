class RandomAgent:
    def __init__(self, env):
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # Track training stats
        self.curr_episode_length = 0
        self.episodes_lengths = []

    def observe(self, state, action, next_state, reward, terminated, truncated):
        self.curr_episode_length += 1

        # End of episode
        if terminated or truncated:
            self.episodes_lengths.append(self.curr_episode_length)
            self.curr_episode_length = 0

    def select_action(self, state):
        return self.action_space.sample()  # take a random action

    def update(self):
        pass

    def save(self):
        pass

    def get_training_dict(self):
        return {
            "episodes_lengths": self.episodes_lengths,
        }
