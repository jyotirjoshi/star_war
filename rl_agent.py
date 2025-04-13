import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging
import time
import json
import gc
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RL_Agent")

# Optimize TensorFlow performance
tf.config.optimizer.set_jit(False)  # Disable XLA which can cause freezes
tf.keras.mixed_precision.set_global_policy('float32')  # Use float32 for stability

# Set memory growth for GPU
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logger.info(f"GPU enabled with memory growth: {physical_devices}")
    else:
        logger.info("Running on CPU")
except Exception as e:
    logger.warning(f"Error configuring GPU: {e}")


class DQNAgent:
    """Optimized Deep Q-Network agent for reinforcement learning"""

    def __init__(self, state_size, action_size):
        """Initialize the agent with optimized parameters

        Args:
            state_size: Dimension of state (int)
            action_size: Dimension of action (int)
        """
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters - optimized for stability and performance
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.99  # Slower decay for more exploration
        self.learning_rate = 0.001  # Learning rate
        self.batch_size = 32  # Smaller batch size to prevent freezing
        self.train_start = 100  # Start training after fewer experiences
        self.update_target_freq = 5  # Update target model every n episodes

        # Create memory buffer - smaller size to reduce memory usage
        self.memory = deque(maxlen=1000)

        # Create the models with simplified architecture
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        logger.info(f"DQN Agent initialized - State Size: {state_size}, Action Size: {action_size}")

    def _build_model(self):
        """Build a simplified neural network model"""
        model = Sequential([
            Dense(32, activation='relu', input_dim=self.state_size),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Update target model weights from training model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Choose an action based on the state

        Args:
            state: Current state
            training: Whether to use epsilon-greedy policy

        Returns:
            selected action
        """
        if training and np.random.rand() <= self.epsilon:
            # Random action (exploration)
            return random.randrange(self.action_size)

        # Exploit learned policy
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def train(self):
        """Train the model on a batch from memory with memory optimizations"""
        if len(self.memory) < self.train_start:
            return

        # Sample a batch from memory
        minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))

        # Extract components
        states = np.array([data[0] for data in minibatch])
        actions = np.array([data[1] for data in minibatch])
        rewards = np.array([data[2] for data in minibatch])
        next_states = np.array([data[3] for data in minibatch])
        dones = np.array([data[4] for data in minibatch])

        # Calculate targets - process in smaller batches if needed
        targets = self.model.predict(states, verbose=0)
        targets_next = self.target_model.predict(next_states, verbose=0)

        for i in range(len(minibatch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(targets_next[i])

        # Train in smaller batches for better performance
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=min(16, len(minibatch)))

        # Collect garbage periodically
        if random.random() < 0.05:
            gc.collect()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """Save the model to disk"""
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path):
        """Load the model from disk"""
        try:
            self.model = load_model(path)
            self.target_model = load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")


class TradingEnv:
    """Reinforcement learning trading environment - optimized for stability"""

    def __init__(self, data, initial_balance=100000, max_position=5, window_size=10):
        """Initialize the environment with optimized parameters

        Args:
            data: Historical price data (pandas DataFrame)
            initial_balance: Starting account balance
            max_position: Maximum position size
            window_size: Number of past observations to include in state (reduced for efficiency)
        """
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Input data missing required column: {col}")

        self.data = data
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.window_size = window_size

        # Define action space and observation space
        self.action_space = 3  # 0=hold, 1=buy, 2=sell

        # Define observation space with correct dimensions
        observation_dim = window_size + 3  # price history + balance + position + PnL

        # Custom observation space class
        class ObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        self.observation_space = ObservationSpace(shape=(observation_dim,))

        # Scaling factors for normalization
        self.price_scale = 1000.0  # Scale factor for prices

        # Initialize price tracking
        self.current_price = 0
        self.last_price = 0

        # Reset environment
        self.reset()

        logger.info(f"Trading environment initialized - Data shape: {data.shape}, Window size: {window_size}")

    def reset(self):
        """Reset the environment to initial state"""
        # Start after window size to have enough history
        self.current_step = self.window_size

        # Reset account state
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.done = False

        # Set initial prices
        self.current_price = self.data.iloc[self.current_step]['close']
        self.last_price = self.data.iloc[self.current_step - 1]['close']

        # Get initial observation
        return self._get_observation()

    def step(self, action):
        """Take a step in the environment based on the action

        Args:
            action: Integer representing action (0=hold, 1=buy, 2=sell)

        Returns:
            observation, reward, done, info
        """
        # Save current state for reward calculation
        old_balance = self.balance
        old_position = self.position

        # Update pricing information
        self.last_price = self.current_price
        self.current_price = self.data.iloc[self.current_step]['close']

        # Process action (0=hold, 1=buy, 2=sell)
        position_change = 0
        if action == 1:  # Buy/increase position
            if self.position < self.max_position:
                position_change = 1
        elif action == 2:  # Sell/decrease position
            if self.position > -self.max_position:
                position_change = -1

        # Execute trade if any
        if position_change != 0:
            # Calculate trade cost and commission
            trade_cost = position_change * self.current_price * 20  # 20 points multiplier for NQ
            commission = abs(position_change) * 2.5  # $2.50 per contract

            # Update balance
            self.balance -= trade_cost + commission

            # Record trade with simplified structure
            self.trades.append({
                'step': self.current_step,
                'action': 1 if position_change > 0 else 2,
                'price': float(self.current_price),
                'cost': float(trade_cost + commission)
            })

            # Update position
            self.position += position_change

        # Move to next step
        self.current_step += 1

        # Check if done
        self.done = (self.current_step >= len(self.data) - 1)

        # Calculate reward
        reward = self._calculate_reward(old_balance, old_position)

        # Return observation, reward, done, info
        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        """Create the observation (state) with consistent dimensions"""
        try:
            # Get price history (safely handling index boundaries)
            start_idx = max(0, self.current_step - self.window_size)
            end_idx = self.current_step
            price_history = self.data.iloc[start_idx:end_idx]

            # Ensure we have enough price history
            if len(price_history) < self.window_size:
                # Pad with the first price if needed
                first_price = price_history.iloc[0]['close'] if not price_history.empty else self.current_price
                padding_size = self.window_size - len(price_history)
                padding = np.full(padding_size, first_price)
                close_prices = np.concatenate([padding, price_history['close'].values])
            else:
                close_prices = price_history['close'].values

            # Normalize price data
            close_prices = close_prices / self.price_scale

            # Create observation with price history and account state
            obs = np.concatenate([
                close_prices,
                [
                    self.balance / self.initial_balance,  # Normalized balance
                    self.position / self.max_position,  # Normalized position
                    self._calculate_unrealized_pnl() / self.initial_balance  # Normalized unrealized P&L
                ]
            ])

            # Ensure observation has correct dimension
            assert len(
                obs) == self.window_size + 3, f"Observation has wrong dimension: {len(obs)} vs expected {self.window_size + 3}"

            return obs

        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            # Return safe fallback observation
            return np.zeros(self.window_size + 3)

    def _calculate_reward(self, old_balance, old_position):
        """Calculate reward based on portfolio change"""
        try:
            # Calculate current portfolio value
            current_portfolio = self.balance + self._calculate_unrealized_pnl()

            # Calculate old portfolio value
            old_price = self.last_price
            old_unrealized_pnl = old_position * (old_price - self._get_entry_price()) * 20
            old_portfolio = old_balance + old_unrealized_pnl

            # Base reward is the change in portfolio value (scaled)
            portfolio_change = current_portfolio - old_portfolio
            reward = portfolio_change / 100  # Scale rewards to reasonable range

            # Add penalty for large drawdowns
            if current_portfolio < self.initial_balance * 0.95:
                reward -= 1.0  # Penalty for significant drawdown

            # Small penalty for holding positions to encourage efficient use of capital
            if self.position != 0:
                reward -= 0.01 * abs(self.position)

            return reward

        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def _calculate_unrealized_pnl(self):
        """Calculate unrealized profit/loss based on current position and price"""
        if self.position == 0:
            return 0

        entry_price = self._get_entry_price()

        # Handle possible division by zero or other errors
        if entry_price <= 0:
            return 0

        return (self.current_price - entry_price) * self.position * 20  # 20 is multiplier for NQ

    def _get_entry_price(self):
        """Calculate average entry price from trades - optimized for the actual trade structure"""
        if self.position == 0 or not self.trades:
            return 0

        # Direction of current position
        is_long = self.position > 0

        # Find trades that match our current direction using 'action' key
        # action 1 = buy/long, action 2 = sell/short
        matching_trades = [t for t in self.trades if
                           (t['action'] == 1 and is_long) or
                           (t['action'] == 2 and not is_long)]

        if not matching_trades:
            return self.current_price  # Fallback to current price

        # Use last matching trade's price as entry price (simplified)
        return matching_trades[-1]['price']


class RLTrainer:
    """Class to train an RL agent on trading data - optimized for performance"""

    def __init__(self, data, window_size=10):
        """Initialize the trainer

        Args:
            data: Historical price data
            window_size: Window of past observations for state (reduced for performance)
        """
        # Progress output
        print(f"Initializing RL trainer with data shape: {data.shape}")

        # Create environment
        self.env = TradingEnv(data, window_size=window_size)

        # Get state size from observation_space
        state_size = self.env.observation_space.shape[0]

        # Get action size
        action_size = self.env.action_space

        print(f"Creating DQN agent with state_size={state_size}, action_size={action_size}")

        # Create DQN agent
        self.agent = DQNAgent(state_size, action_size)

        # Training metrics
        self.rewards = []
        self.portfolio_values = []

    def train(self, episodes=100, save_path=None):
        """Train the agent with improved monitoring and error handling

        Args:
            episodes: Number of training episodes
            save_path: Path to save the trained model

        Returns:
            Training metrics
        """
        # Progress tracking
        logger.info(f"Starting RL training for {episodes} episodes")
        print(f"Training for {episodes} episodes...")
        start_time = time.time()

        episode_rewards = []
        episode_portfolio_values = []

        try:
            for e in range(episodes):
                episode_start_time = time.time()

                # Reset environment
                state = self.env.reset()

                # Initialize episode metrics
                episode_reward = 0
                step_count = 0

                # Print initial state info for debugging
                if e == 0:
                    print(f"Initial state shape: {state.shape}")

                while not self.env.done:
                    # Get action
                    action = self.agent.act(state)

                    # Take action
                    next_state, reward, done, info = self.env.step(action)

                    # Store experience in memory
                    self.agent.remember(state, action, reward, next_state, done)

                    # Update state
                    state = next_state

                    # Train agent
                    self.agent.train()

                    # Update episode reward
                    episode_reward += reward
                    step_count += 1

                    # Periodic progress update within episode
                    if step_count % 500 == 0:
                        print(f"Episode {e + 1}/{episodes}, Step {step_count}, Reward: {episode_reward:.2f}")

                        # Force garbage collection periodically
                        gc.collect()

                # After episode ends
                episode_rewards.append(episode_reward)
                final_portfolio = self.env.balance + self.env.position * self.env.current_price * 20
                episode_portfolio_values.append(final_portfolio)

                # Update target network
                if e % self.agent.update_target_freq == 0:
                    self.agent.update_target_model()

                # Log progress with episode duration
                episode_time = time.time() - episode_start_time
                if (e + 1) % max(1, episodes // 20) == 0 or e == 0:
                    avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
                    avg_portfolio = np.mean(episode_portfolio_values[-10:]) if len(
                        episode_portfolio_values) >= 10 else final_portfolio
                    logger.info(
                        f"Episode {e + 1}/{episodes} | Time: {episode_time:.1f}s | Reward: {episode_reward:.2f} | Portfolio: ${final_portfolio:.2f} | Epsilon: {self.agent.epsilon:.4f}")
                    print(
                        f"Episode {e + 1}/{episodes} | Time: {episode_time:.1f}s | Avg Reward: {avg_reward:.2f} | Portfolio: ${final_portfolio:.2f}")

                # Intermediate save
                if save_path and episodes > 10 and (e + 1) % (episodes // 5) == 0:
                    intermediate_path = f"{os.path.splitext(save_path)[0]}_episode_{e + 1}.h5"
                    self.agent.save(intermediate_path)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user, saving current model...")
            if save_path:
                self.agent.save(f"{os.path.splitext(save_path)[0]}_interrupted.h5")

        except Exception as e:
            logger.error(f"Error during training: {e}")
            print(f"Error during training: {e}")
            if save_path:
                self.agent.save(f"{os.path.splitext(save_path)[0]}_error_recovery.h5")
            raise

        # Save final model
        if save_path:
            model_dir = os.path.dirname(save_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
            self.agent.save(save_path)

        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        print(f"Training completed in {training_time:.2f} seconds")

        # Save training metrics
        training_metrics = {
            'episode_rewards': episode_rewards,
            'episode_portfolio_values': episode_portfolio_values,
            'initial_balance': self.env.initial_balance,
            'final_portfolio': episode_portfolio_values[-1] if episode_portfolio_values else 0,
            'return_pct': ((episode_portfolio_values[
                                -1] if episode_portfolio_values else 0) - self.env.initial_balance) / self.env.initial_balance * 100,
            'training_time': training_time
        }

        return training_metrics

    def test(self, model_path=None):
        """Test the trained agent"""
        # Load model if path provided
        if model_path:
            self.agent.load(model_path)

        # Reset environment
        state = self.env.reset()

        # Test metrics
        rewards = []
        portfolio_values = [self.env.initial_balance]
        actions_taken = []

        # Disable exploration
        self.agent.epsilon = 0.0

        print("Starting test run...")

        try:
            # Run test episode
            while not self.env.done:
                # Get action
                action = self.agent.act(state, training=False)

                # Take action
                next_state, reward, done, info = self.env.step(action)

                # Update state
                state = next_state

                # Record metrics
                rewards.append(reward)
                portfolio_values.append(self.env.balance + self.env.position * self.env.current_price * 20)
                actions_taken.append(action)

                # Progress update
                if len(actions_taken) % 500 == 0:
                    print(f"Test step {len(actions_taken)}, Portfolio: ${portfolio_values[-1]:.2f}")

            # Calculate test metrics
            test_metrics = {
                'total_reward': sum(rewards),
                'initial_balance': self.env.initial_balance,
                'final_portfolio': portfolio_values[-1],
                'return_pct': (portfolio_values[-1] - self.env.initial_balance) / self.env.initial_balance * 100,
                'actions': {
                    'hold': actions_taken.count(0),
                    'buy': actions_taken.count(1),
                    'sell': actions_taken.count(2)
                }
            }

            print(
                f"Test completed - Return: {test_metrics['return_pct']:.2f}%, Final: ${test_metrics['final_portfolio']:.2f}")

            return test_metrics, portfolio_values

        except Exception as e:
            logger.error(f"Error during testing: {e}")
            print(f"Error during testing: {e}")
            return None, None

    def plot_results(self, portfolio_values, save_path=None):
        """Plot training results"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(portfolio_values)
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Step')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)

            if save_path:
                plt.savefig(save_path)
                print(f"Results plot saved to {save_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            print(f"Error plotting results: {e}")


# Debug utility to run a short training session
def debug_train():
    """Run a minimal training session to diagnose issues"""
    print("Starting debug training with synthetic data...")

    # Create minimal synthetic data
    dates = pd.date_range('2025-01-01', periods=500)
    data = pd.DataFrame({
        'open': np.random.normal(100, 5, 500),
        'high': np.random.normal(105, 5, 500),
        'low': np.random.normal(95, 5, 500),
        'close': np.random.normal(100, 5, 500),
        'volume': np.random.normal(1000, 200, 500)
    }, index=dates)

    # Use smaller window and fewer episodes
    print("Creating trainer...")
    trainer = RLTrainer(data, window_size=5)

    print("Starting training with 3 episodes...")
    training_results = trainer.train(episodes=3)

    print("Training completed!")
    print(f"Final reward: {training_results['episode_rewards'][-1]}")
    return training_results


if __name__ == "__main__":
    # Test the module with debug training
    debug_train()