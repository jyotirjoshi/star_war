# Add these imports at the top of the file
import gc
import os
import psutil


def process_memory_usage():
    """Get current memory usage of process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class DQNAgent:
    # Add this to your existing DQNAgent class
    def __init__(self, state_size, action_size):
        # Existing initialization code...

        # Memory optimization settings
        tf.config.optimizer.set_jit(False)  # Disable XLA which can cause freezes
        self.batch_size = 32  # Smaller batch size

        # Simplified model for better performance
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Build a simpler neural network model"""
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))  # Smaller first layer
        model.add(Dense(16, activation='relu'))  # Smaller second layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def train(self):
        """Train with improved memory management"""
        if len(self.memory) < self.train_start:
            return

        # Sample a smaller batch for better performance
        minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))

        # Extract components
        states = np.array([data[0] for data in minibatch])
        actions = np.array([data[1] for data in minibatch])
        rewards = np.array([data[2] for data in minibatch])
        next_states = np.array([data[3] for data in minibatch])
        dones = np.array([data[4] for data in minibatch])

        # Explicitly free memory
        targets = self.model.predict(states, verbose=0)

        # Predict in smaller batches if needed
        targets_next = self.target_model.predict(next_states, verbose=0)

        for i in range(len(minibatch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(targets_next[i])

        # Train in smaller batches if needed
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=min(16, len(minibatch)))

        # Force garbage collection occasionally
        if random.random() < 0.01:  # Do this occasionally
            gc.collect()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay