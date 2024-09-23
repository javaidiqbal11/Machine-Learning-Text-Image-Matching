# py file used for task 2 define din the requirements 
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install gym')
get_ipython().system('pip install stable-baselines3[extra]')
get_ipython().system('pip install gym-super-mario-bros')


# # DQN (Deep Q-Network)

# In[ ]:


import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from torchvision import transforms
from PIL import Image
import time

# Setup the environment
env = gym_super_mario_bros.make('SuperMarioBros2-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# CNN definition
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

def preprocess(state):
    state = Image.fromarray(state)
    state = transform(state).unsqueeze(0)
    return state

# Setup hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(1, len(SIMPLE_MOVEMENT)).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.00025)
memory = deque(maxlen=10000)
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

# Training variables
num_episodes = 10
target_update = 10
batch_size = 32
gamma = 0.99
steps_done = 0

# Metrics
total_rewards = []
total_scores = []
total_steps = []

# Training loop
training_start = time.time()
for episode in range(num_episodes):
    state = env.reset()
    state = preprocess(state)
    total_reward = 0
    steps = 0
    score = 0

    while True:
        steps_done += 1
        steps += 1

        if random.random() > epsilon:
            with torch.no_grad():
                action = policy_net(state.to(device)).max(1)[1].view(1, 1).item()
        else:
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        next_state = preprocess(next_state)
        memory.append((state, action, reward, next_state, done))
        state = next_state

        total_reward += reward
        score = info['score']

        if done:
            break

    total_rewards.append(total_reward)
    total_scores.append(score)
    total_steps.append(steps)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 100 == 0:
        print(f'Episode {episode}: Average Reward: {np.mean(total_rewards[-100:])}, Average Score: {np.mean(total_scores[-100:])}')

training_end = time.time()
training_time = training_end - training_start

# Evaluation
average_reward = np.mean(total_rewards)
average_score = np.mean(total_scores)
average_steps = np.mean(total_steps)

print(f"Average Reward: {average_reward}")
print(f"Average Game Score: {average_score}")
print(f"Average Steps Per Episode: {average_steps}")
print(f"Training Time: {training_time} seconds")


# In[ ]:


import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from torchvision import transforms
from PIL import Image
import time

# Setup the environment
env = gym_super_mario_bros.make('SuperMarioBros2-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

def ppo_update(policy_net, optimizer, states, actions, log_probs_old, returns, advantages, epsilon_clip=0.2, c1=0.5, c2=0.01):
    for _ in range(4):  # Perform multiple updates using the same batch
        action_probs, state_values = policy_net(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        ratio = (log_probs - log_probs_old).exp()

        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        critic_loss = F.mse_loss(state_values.squeeze(-1), returns)

        # Entropy bonus
        entropy_bonus = dist.entropy().mean()

        # Total loss
        loss = actor_loss + c1 * critic_loss - c2 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def preprocess(state):
    state = Image.fromarray(state)
    state = transform(state).unsqueeze(0)
    return state

# Setup hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(1, len(SIMPLE_MOVEMENT)).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.00025)
memory = deque(maxlen=10000)
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

# Training variables
num_episodes = 10
target_update = 10
batch_size = 32
gamma = 0.99
steps_done = 0

# Metrics
total_rewards = []
total_scores = []
total_steps = []

# Training loop
training_start = time.time()
for episode in range(num_episodes):
    state = env.reset()
    state = preprocess(state)
    total_reward = 0
    steps = 0
    score = 0

    while True:
        steps_done += 1
        steps += 1

        if random.random() > epsilon:
            with torch.no_grad():
                action = policy_net(state.to(device)).max(1)[1].view(1, 1).item()
        else:
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        next_state = preprocess(next_state)
        memory.append((state, action, reward, next_state, done))
        state = next_state

        total_reward += reward
        score = info['score']

        if done:
            break

    total_rewards.append(total_reward)
    total_scores.append(score)
    total_steps.append(steps)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 100 == 0:
        print(f'Episode {episode}: Average Reward: {np.mean(total_rewards[-100:])}, Average Score: {np.mean(total_scores[-100:])}')

training_end = time.time()
training_time = training_end - training_start

# Evaluation
average_reward = np.mean(total_rewards)
average_score = np.mean(total_scores)
average_steps = np.mean(total_steps)

print(f"Average Reward: {average_reward}")
print(f"Average Game Score: {average_score}")
print(f"Average Steps Per Episode: {average_steps}")
print(f"Training Time: {training_time} seconds")


# In[9]:


import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from torchvision import transforms
from PIL import Image
import time

# Setup the environment
env = gym_super_mario_bros.make('SuperMarioBros2-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

class ActorCritic(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)

        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))

        policy = F.softmax(self.policy(x), dim=-1)
        value = self.value(x)

        return policy, value
def preprocess(state):
    state = Image.fromarray(state)
    state = transform(state).unsqueeze(0)
    return state

# Setup hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(1, len(SIMPLE_MOVEMENT)).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.00025)
memory = deque(maxlen=10000)
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

# Training variables
num_episodes = 10
target_update = 10
batch_size = 32
gamma = 0.99
steps_done = 0

# Metrics
total_rewards = []
total_scores = []
total_steps = []

# Training loop
training_start = time.time()
for episode in range(num_episodes):
    state = env.reset()
    state = preprocess(state)
    total_reward = 0
    steps = 0
    score = 0

    while True:
        steps_done += 1
        steps += 1

        if random.random() > epsilon:
            with torch.no_grad():
                action = policy_net(state.to(device)).max(1)[1].view(1, 1).item()
        else:
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        next_state = preprocess(next_state)
        memory.append((state, action, reward, next_state, done))
        state = next_state

        total_reward += reward
        score = info['score']

        if done:
            break

    total_rewards.append(total_reward)
    total_scores.append(score)
    total_steps.append(steps)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 100 == 0:
        print(f'Episode {episode}: Average Reward: {np.mean(total_rewards[-100:])}, Average Score: {np.mean(total_scores[-100:])}')

training_end = time.time()
training_time = training_end - training_start

# Evaluation
average_reward = np.mean(total_rewards)
average_score = np.mean(total_scores)
average_steps = np.mean(total_steps)

print(f"Average Reward: {average_reward}")
print(f"Average Game Score: {average_score}")
print(f"Average Steps Per Episode: {average_steps}")
print(f"Training Time: {training_time} seconds")

