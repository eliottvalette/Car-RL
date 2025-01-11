### Training Script (race_train.py) ###
from race_game import CarRacingGame
from race_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import torch

# Hyperparameters
EPISODES = 1000
STATE_SIZE = 16  
ACTION_SIZE = 4

# Constants
LOAD_MODEL = False

# Initialize environment and agent
env = CarRacingGame()
agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

# Tracking variables
scores = []
epsilons = []

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load model if specified
if LOAD_MODEL:
    agent.model.load_state_dict(torch.load('models/model.pth'))

for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
        step_count += 1

        if episode%10 == 0:
            env.render(fps = 1000)  # Optional, for visualization

    agent.decay_epsilon()

    scores.append(total_reward)
    epsilons.append(agent.epsilon)

    print(f"Episode: {episode + 1}/{EPISODES}")
    print(f"Steps: {step_count}")
    print(f"Laps: {env.laps}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Epsilon: {agent.epsilon:.2f}")
    print("-" * 40)

    # Update target network every 10 episodes
    if episode % 10 == 0:
        agent.update_target_model()

    if episode % 100 == 0:
        print(f"Saving model at episode {episode}")
        torch.save(agent.model.state_dict(), f'models/model_{episode}.pth')

# save model
torch.save(agent.model.state_dict(), 'models/model.pth')

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(scores, label="Scores")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.legend()
plt.show()

# Clean up
env.close()
