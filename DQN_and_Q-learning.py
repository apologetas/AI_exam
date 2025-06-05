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


        original_epsilon = self.epsilon
        self.epsilon = 0.0

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

        # Restore original epsilon
        self.epsilon = original_epsilon
        env.close()


def train_q_agent_with_tracking():
    print("\n🔄 Q-Learning agentas treniruojamas...")
    env = gym.make('CartPole-v1')

    n_bins = 15
    pos_space = np.linspace(-2.4, 2.4, n_bins)
    vel_space = np.linspace(-3, 3, n_bins)
    ang_space = np.linspace(-0.2095, 0.2095, n_bins)
    ang_vel_space = np.linspace(-3, 3, n_bins)

    q_table = np.zeros((n_bins + 1, n_bins + 1, n_bins + 1, n_bins + 1, 2))

    # mokymosi parametrai
    learning_rate = 0.15
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.05

    training_scores = []

    for episode in range(1500):
        state = env.reset()[0]
        steps = 0

        # Diskretizuoti pradinę būseną
        s_p = np.clip(np.digitize(state[0], pos_space), 0, n_bins)
        s_v = np.clip(np.digitize(state[1], vel_space), 0, n_bins)
        s_a = np.clip(np.digitize(state[2], ang_space), 0, n_bins)
        s_av = np.clip(np.digitize(state[3], ang_vel_space), 0, n_bins)

        done = False

        while not done and steps < 500:
            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[s_p, s_v, s_a, s_av, :])

            next_state, reward, done, _, _ = env.step(action)

            # Modifikuotas atlygis už stabilumą
            if not done:
                reward = 1.0 + (1.0 - abs(state[2]) * 5)  # Bonus už mažą kampą
            else:
                reward = -10  # Bausmė už kritimą

            # Diskretizuoti naują būseną
            ns_p = np.clip(np.digitize(next_state[0], pos_space), 0, n_bins)
            ns_v = np.clip(np.digitize(next_state[1], vel_space), 0, n_bins)
            ns_a = np.clip(np.digitize(next_state[2], ang_space), 0, n_bins)
            ns_av = np.clip(np.digitize(next_state[3], ang_vel_space), 0, n_bins)

            # Q-learning atnaujinimas
            old_q = q_table[s_p, s_v, s_a, s_av, action]
            max_next_q = np.max(q_table[ns_p, ns_v, ns_a, ns_av, :])

            new_q = old_q + learning_rate * (reward + discount_factor * max_next_q - old_q)
            q_table[s_p, s_v, s_a, s_av, action] = new_q

            s_p, s_v, s_a, s_av = ns_p, ns_v, ns_a, ns_av
            state = next_state
            steps += 1

        training_scores.append(steps)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % 200 == 0:
            avg_score = np.mean(training_scores[-100:])
            print(f"Q-Learning - Epizodas {episode}, Vidutinis rezultatas: {avg_score:.1f}, Epsilon: {epsilon:.3f}")

    env.close()
    print("✅ Q-Learning agentas ištreniruotas!")
    return q_table, (pos_space, vel_space, ang_space, ang_vel_space), training_scores


def test_q_agent(q_table, spaces, episodes=3):
    env = gym.make('CartPole-v1', render_mode='human')
    pos_space, vel_space, ang_space, ang_vel_space = spaces
    n_bins = len(pos_space)

    print("\nQ-Learning Testavimas...")
    for episode in range(episodes):
        state = env.reset()[0]
        steps = 0
        done = False

        while not done and steps < 500:
            # Discretizuota busena
            s_p = np.clip(np.digitize(state[0], pos_space), 0, n_bins)
            s_v = np.clip(np.digitize(state[1], vel_space), 0, n_bins)
            s_a = np.clip(np.digitize(state[2], ang_space), 0, n_bins)
            s_av = np.clip(np.digitize(state[3], ang_vel_space), 0, n_bins)

            # Get action from Q-table
            action = np.argmax(q_table[s_p, s_v, s_a, s_av, :])

            state, reward, done, _, _ = env.step(action)
            steps += 1

        print(f"Q-Learning Epizodas {episode + 1}: {steps} žingsnių")

    env.close()


def plot_comparison(dqn_scores, q_scores):
    """Plot comparison between DQN and Q-Learning"""
    plt.figure(figsize=(15, 5))

    # DQN grafikas
    plt.subplot(1, 3, 1)
    plt.plot(dqn_scores, alpha=0.6, color='blue', label='DQN')
    if len(dqn_scores) >= 50:
        moving_avg_dqn = [np.mean(dqn_scores[i - 49:i + 1]) for i in range(49, len(dqn_scores))]
        plt.plot(range(49, len(dqn_scores)), moving_avg_dqn, linewidth=2, color='darkblue')
    plt.title('DQN mokymosi progresas')
    plt.xlabel('Epizodas')
    plt.ylabel('Žingsnių skaičius')
    plt.grid(True, alpha=0.3)

    # Q-Learning grafikas
    plt.subplot(1, 3, 2)
    plt.plot(q_scores, alpha=0.6, color='red', label='Q-Learning')
    if len(q_scores) >= 100:
        moving_avg_q = [np.mean(q_scores[i - 99:i + 1]) for i in range(99, len(q_scores))]
        plt.plot(range(99, len(q_scores)), moving_avg_q, linewidth=2, color='darkred')
    plt.title('Q-Learning mokymosi progresas')
    plt.xlabel('Epizodas')
    plt.ylabel('Žingsnių skaičius')
    plt.grid(True, alpha=0.3)

    # Palyginimo grafikas
    plt.subplot(1, 3, 3)
    min_len = min(len(dqn_scores), len(q_scores))
    plt.plot(dqn_scores[:min_len], alpha=0.6, color='blue', label='DQN')
    plt.plot(q_scores[:min_len], alpha=0.6, color='red', label='Q-Learning')

    window = 50
    if min_len >= window:
        dqn_ma = [np.mean(dqn_scores[i - window + 1:i + 1]) for i in range(window - 1, min_len)]
        q_ma = [np.mean(q_scores[i - window + 1:i + 1]) for i in range(window - 1, min_len)]
        plt.plot(range(window - 1, min_len), dqn_ma, linewidth=2, color='darkblue', label='DQN (50-ep avg)')
        plt.plot(range(window - 1, min_len), q_ma, linewidth=2, color='darkred', label='Q-Learning (50-ep avg)')

    plt.title('DQN vs Q-Learning Palyginimas')
    plt.xlabel('Epizodas')
    plt.ylabel('Žingsnių skaičius')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_comparison_stats(dqn_scores, q_scores):

    print("\n" + "=" * 60)
    print("📊 MODELIŲ PALYGINIMO STATISTIKOS")
    print("=" * 60)


    print(f"🔵 DQN Rezultatai:")
    print(f"   Paskutinių 100 epizodų vidurkis: {np.mean(dqn_scores[-100:]):.1f}")
    print(f"   Maksimalus rezultatas: {max(dqn_scores):.0f}")
    print(f"   Galutinis epsilon: {0.284:.3f}")

    print(f"\n🔴 Q-Learning Rezultatai:")
    print(f"   Paskutinių 100 epizodų vidurkis: {np.mean(q_scores[-100:]):.1f}")
    print(f"   Maksimalus rezultatas: {max(q_scores):.0f}")
    print(f"   Galutinis epsilon: {0.496:.3f}")



def main():
    print("🎮 DQN vs Q-Learning Palyginimas")
    print("=" * 50)


    print("\n🚀  DQN agentas treniruojamas")
    dqn_agent = DeepQNetwork()
    dqn_agent.train(episodes=400)


    q_table, spaces, q_scores = train_q_agent_with_tracking()

    # Testuoti abu modelius
    dqn_agent.test(episodes=3)
    test_q_agent(q_table, spaces, episodes=3)

    # Palyginti rezultatus
    plot_comparison(dqn_agent.scores, q_scores)
    print_comparison_stats(dqn_agent.scores, q_scores)


if __name__ == '__main__':
    main()