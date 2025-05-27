import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import logging
from .indicators import TechnicalIndicators

logger = logging.getLogger('rl_predictor')

class RLPredictor:
    def __init__(self, state_size=15, action_size=3, memory_size=10000, gamma=0.95, epsilon=1.0):
        """Initialize RL predictor with Deep Q-Learning
        
        Args:
            state_size (int): Number of features in state space
            action_size (int): Number of possible actions (buy, sell, hold)
            memory_size (int): Size of replay memory
            gamma (float): Discount factor for future rewards
            epsilon (float): Initial exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_counter = 0
        self.model_path = os.getenv('RL_PREDICTOR_PATH', 'models/rl_predictor')
        
    def _build_model(self):
        """Build deep Q-learning model"""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def _get_state(self, df, idx):
        """Create state vector from market data"""
        state = []
        
        # Price features
        state.extend([
            df['close'].iloc[idx],
            df['high'].iloc[idx],
            df['low'].iloc[idx],
            df['volume'].iloc[idx]
        ])
        
        # Technical indicators
        state.extend([
            df['rsi'].iloc[idx],
            df['macd'].iloc[idx],
            df['adx'].iloc[idx],
            df['bb_width'].iloc[idx],
            df['atr_pct'].iloc[idx],
            df['momentum'].iloc[idx],
            df['stoch_rsi_k'].iloc[idx]
        ])
        
        # Regime features if available
        if 'regime' in df.columns:
            state.extend([df[f'regime_{i}'].iloc[idx] for i in range(3)])
        else:
            state.extend([0, 0, 0])
        
        # Normalize state
        state = np.array(state)
        state = (state - np.mean(state)) / (np.std(state) + 1e-8)
        
        return state
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=32):
        """Train model on experiences from replay memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # Double DQN update
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        max_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        target_q = rewards + self.gamma * next_q[np.arange(batch_size), max_actions] * (1 - dones)
        
        target = current_q.copy()
        target[np.arange(batch_size), actions] = target_q
        
        self.model.fit(states, target, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.update_target_counter += 1
        if self.update_target_counter % 100 == 0:
            self.target_model.set_weights(self.model.get_weights())
    
    def train(self, df, episodes=100, batch_size=32):
        """Train the RL agent"""
        try:
            logger.info("Training RL agent...")
            
            # Calculate technical indicators if not present
            df = TechnicalIndicators.calculate_all_indicators(df)
            
            best_reward = float('-inf')
            episode_rewards = []
            
            for episode in range(episodes):
                total_reward = 0
                state = self._get_state(df, 0)
                
                for t in range(1, len(df)):
                    # Choose and perform action
                    action = self.act(state)
                    next_state = self._get_state(df, t)
                    
                    # Calculate reward based on position and price movement
                    price_change = df['close'].iloc[t] / df['close'].iloc[t-1] - 1
                    if action == 0:  # Buy
                        reward = price_change
                    elif action == 1:  # Sell
                        reward = -price_change
                    else:  # Hold
                        reward = abs(price_change) * 0.1  # Small reward for staying out of volatile markets
                    
                    done = t == len(df) - 1
                    total_reward += reward
                    
                    # Store experience
                    self.remember(state, action, reward, next_state, done)
                    
                    # Train on batch
                    self.replay(batch_size)
                    
                    state = next_state
                    
                    if done:
                        episode_rewards.append(total_reward)
                        avg_reward = np.mean(episode_rewards[-100:])
                        logger.info(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")
                        
                        if total_reward > best_reward:
                            best_reward = total_reward
                            # Save best model
                            if not os.path.exists(os.path.dirname(self.model_path)):
                                os.makedirs(os.path.dirname(self.model_path))
                            self.model.save(self.model_path)
            
            logger.info("RL training completed")
            return episode_rewards
            
        except Exception as e:
            logger.error(f"Error training RL agent: {str(e)}")
            return None
    
    def predict(self, df):
        """Make predictions using trained RL agent"""
        try:
            # Load model if not loaded
            if not os.path.exists(self.model_path):
                logger.error("No trained model found. Please run train() first.")
                return None
            
            # Calculate technical indicators
            df = TechnicalIndicators.calculate_all_indicators(df)
            
            # Get current state
            current_state = self._get_state(df, -1)
            
            # Get Q-values for all actions
            q_values = self.model.predict(current_state.reshape(1, -1), verbose=0)[0]
            
            # Convert Q-values to probabilities using softmax
            probabilities = tf.nn.softmax(q_values).numpy()
            
            return {
                'action_probabilities': probabilities,
                'q_values': q_values,
                'recommended_action': np.argmax(q_values)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None 