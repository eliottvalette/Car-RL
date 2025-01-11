### Training Script (race_train.py) ###
from race_game import CarRacingGame
from race_agent import CarAgent
import numpy as np
import matplotlib.pyplot as plt
import torch

# Hyperparameters
EPISODES = 1000
STATE_SIZE = 16  
ACTION_SIZE = 4

# Constants
LOAD_MODEL = True

# Initialize environment and agent
env = CarRacingGame()
agent = CarAgent(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Tracking variables
scores = []
epsilons = []

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load model if specified
if LOAD_MODEL:
    agent.load("models/model.pth")

for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay()
        state = next_state
        total_reward += reward
        step_count += 1

        if episode%1 == 0:
            env.render(fps = 1000)

    scores.append(total_reward)
    epsilons.append(agent.epsilon)

    print("-" * 40)
    print(f"Episode: {episode + 1}/{EPISODES}")
    print(f"Steps: {step_count}")
    print(f"Laps: {env.laps}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Epsilon: {agent.epsilon:.2f}")
    if loss is not None:
        print(f"Loss: {loss:.4f}")

    # Decay epsilon
    agent.epsilon = 0 # max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    # Save model periodically
    if episode % 100 == 0:
        print(f"Saving model at episode {episode}")
        agent.save(f"models/model_{episode}.pth")

# save model
agent.save("models/model.pth")

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
