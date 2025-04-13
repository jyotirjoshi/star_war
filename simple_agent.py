import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import pandas as pd

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class SimpleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.batch_size = 16
        self.model = self._build_model()

    def _build_model(self):
        # Simple model with minimal layers
        model = Sequential([
            Dense(16, input_dim=self.state_size, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        minibatch = random.sample(self.memory, self.batch_size)

        # Extract components
        states = np.array([data[0] for data in minibatch])
        actions = np.array([data[1] for data in minibatch])
        rewards = np.array([data[2] for data in minibatch])
        next_states = np.array([data[3] for data in minibatch])
        dones = np.array([data[4] for data in minibatch])

        # Calculate targets
        targets = self.model.predict(states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                next_state_values = self.model.predict(np.array([next_states[i]]), verbose=0)[0]
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_state_values)

        # Train
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def test_train():
    """Quick test function to verify agent works"""
    # Generate sample data
    data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(100, 5, 100),
    })

    # Create agent
    agent = SimpleDQNAgent(state_size=13, action_size=3)

    # Fake training loop
    for episode in range(3):
        state = np.random.random(13)  # Fake state with 13 dimensions

        for step in range(50):
            action = agent.act(state)
            next_state = np.random.random(13)
            reward = np.random.random()
            done = False

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state

            if step % 10 == 0:
                print(f"Episode {episode + 1}, Step {step}")

    print("Test completed successfully!")


if __name__ == "__main__":
    test_train()