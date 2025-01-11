### RL Agent Implementation (race_agent.py) ###
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=0.0005, batch_size=128, memory_size=20000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.992

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = "cpu"

        # Model and target model
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(channels, channels),
                    nn.LayerNorm(channels),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(channels, channels),
                    nn.LayerNorm(channels)
                )
            
            def forward(self, x):
                return F.relu(x + self.layers(x))

        class DQNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super().__init__()
                
                self.input_norm = nn.LayerNorm(state_size)
                
                self.feature_extractor = nn.Sequential(
                    nn.Linear(state_size, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                self.residual_blocks = nn.ModuleList([
                    ResidualBlock(256) for _ in range(3)
                ])
                
                self.advantage_stream = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, action_size)
                )
                
                self.value_stream = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1)
                )
                
            def forward(self, x):
                x = self.input_norm(x)
                x = self.feature_extractor(x)
                
                for residual in self.residual_blocks:
                    x = residual(x)
                
                advantage = self.advantage_stream(x)
                value = self.value_stream(x)
                
                # Combine value and advantage (Dueling DQN architecture)
                q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
                return q_values

        return DQNetwork(self.state_size, self.action_size)

    # Add weight initialization method
    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)

            target = self.model(state).clone().detach()

            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    max_next_q = torch.max(self.target_model(next_state))
                target[action] = reward + self.gamma * max_next_q

            states.append(state)
            targets.append(target)

        states = torch.stack(states)
        targets = torch.stack(targets)

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(states), targets)
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

