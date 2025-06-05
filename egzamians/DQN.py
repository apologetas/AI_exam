import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)


class DeepQNetwork:
    def __init__(self):
        self.q_network = DQN()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95

        self.scores = []

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return random.choice([0, 1])

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < 32:
            return

        batch = random.sample(self.memory, 32)
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.q_network(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=1500):
        env = gym.make('CartPole-v1')

        for episode in range(episodes):
            state = env.reset()[0]
            total_reward = 0

            for step in range(500):
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)

                # Mokymasis kas 10 žingsnių
                if step % 10 == 0:
                    self.train_step()

                state = next_state
                total_reward += reward

                if done:
                    break

            self.scores.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if episode % 50 == 0:
                avg_score = np.mean(self.scores[-50:])
                print(f"DQN - Epizodas {episode}, Vidurkis: {avg_score:.1f}, Epsilon: {self.epsilon:.3f}")

        env.close()

    def test(self, episodes=3):
        env = gym.make('CartPole-v1', render_mode='human')

        print("\nDQN Testavimas...")
        for episode in range(episodes):
            state = env.reset()[0]
            total_reward = 0

            for step in range(500):
                action = self.get_action(state)
                state, reward, done, _, _ = env.step(action)
                total_reward += reward

                if done:
                    break

            print(f"DQN Epizodas {episode + 1}: {total_reward} žingsnių")

        env.close()


if __name__ == '__main__':
    print("🚀 DQN Agentas")
    dqn_agent = DeepQNetwork()
    dqn_agent.train(episodes=300)
    dqn_agent.test(episodes=3)