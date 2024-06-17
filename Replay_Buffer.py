class Replay_Buffer():
    def __init__(self,input_shape, n_actions,max_size=int(1e6)):
        self.memory_size = max_size
        self.memory_counter = 0
        self.state = np.zeros((self.memory_size, input_shape))
        self.state_ = np.zeros((self.memory_size, input_shape))
        self.action = np.zeros((self.memory_size, n_actions))
        self.reward = np.zeros(self.memory_size)
        self.done = np.zeros(self.memory_size)

    def add(self, state, action, reward, state_, done):
        index = self.memory_counter % self.memory_size
        self.state[index] = state
        self.state_[index] = state_
        self.action[index] = action
        self.reward[index] = reward
        self.done[index] = done
        self.memory_counter += 1

    def sample(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)
        state = self.state[batch]
        action= self.action[batch]
        reward = self.reward[batch]
        state_ = self.state_[batch]
        done = self.done[batch]
        return state, action, reward, state_, done
