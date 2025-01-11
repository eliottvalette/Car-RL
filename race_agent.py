import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        
        # Actor (Policy) Stream
        self.actor_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )
        
        # Critic (Value) Stream
        self.critic_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Initialize weights using orthogonal initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
            
    def forward(self, state):
        shared_features = self.shared_layers(state)
        
        # Actor output (action probabilities)
        action_probs = F.softmax(self.actor_layers(shared_features), dim=-1)
        
        # Critic output (state value)
        state_value = self.critic_layers(shared_features)
        
        return action_probs, state_value

class CarAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = 0.99
        self.learning_rate = 3e-4
        self.entropy_coefficient = 0.01
        self.value_loss_coefficient = 0.5
        self.max_grad_norm = 0.5
        self.batch_size = 128
        
        # Exploration parameters
        self.epsilon = 0.02
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        
        # Initialize network and optimizer
        self.network = ActorCriticNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=100000)
        
        # Episode tracking
        self.training_step = 0
        
    def remember(self, state, action, reward, next_state, done):
        experience = Experience(
            torch.FloatTensor(state).to(self.device),
            action,
            reward,
            torch.FloatTensor(next_state).to(self.device),
            done
        )
        self.memory.append(experience)
        
    def act(self, state, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            action_probs, _ = self.network(state_tensor)
            
        # During training, sample from probability distribution
        if training:
            action = torch.multinomial(action_probs, 1).item()
        # During evaluation, take the most probable action
        else:
            action = torch.argmax(action_probs).item()
            
        return action
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*batch))
        
        # Convert to tensors
        state_batch = torch.stack(batch.state)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Get current predictions
        action_probs, state_values = self.network(state_batch)
        
        # Get next state values
        with torch.no_grad():
            _, next_state_values = self.network(next_state_batch)
            next_state_values = next_state_values.squeeze(-1)
            
        # Calculate target values using TD(0)
        target_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)
        
        # Calculate advantages
        advantages = target_values - state_values.squeeze(-1)
        
        # Calculate policy loss (using PPO-style objective)
        action_log_probs = torch.log(action_probs + 1e-10)
        selected_action_log_probs = action_log_probs.gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(advantages.detach() * selected_action_log_probs).mean()
        
        # Calculate value loss
        value_loss = F.mse_loss(state_values.squeeze(-1), target_values.detach())
        
        # Calculate entropy bonus (to encourage exploration)
        entropy = -(action_probs * action_log_probs).sum(-1).mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_loss_coefficient * value_loss -
            self.entropy_coefficient * entropy
        )
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.training_step += 1
        
        return total_loss.item()
    
    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']