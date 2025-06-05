import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class CartPoleMDP:

    def __init__(self, n_bins=6, n_episodes=2000):
        print("🎮 CARTPOLE MDP PROJEKTAS")
        self.n_bins = n_bins
        self.env = gym.make('CartPole-v1')

        # MDP KOMPONENTŲ APIBRĖŽIMAS
        self.pos_bins = np.linspace(-2.4, 2.4, n_bins)
        self.vel_bins = np.linspace(-3, 3, n_bins)
        self.ang_bins = np.linspace(-0.2095, 0.2095, n_bins)
        self.ang_vel_bins = np.linspace(-3, 3, n_bins)

        self.n_states = (n_bins + 1) ** 4
        self.n_actions = 2

        # MDP struktūros
        self.transitions = defaultdict(list)
        self.rewards = defaultdict(list)
        self.P = {}  # P(s'|s,a)
        self.R = {}  # R(s,a,s')
        self.V = {}  # V*(s)
        self.policy = {}  # π*(s)

        # APLINKOS TYRINĖJIMAS IR MODELIO KŪRIMAS
        self._explore_environment(n_episodes)
        self._build_transition_probabilities()
        print("✅ MDP SUKURTAS SĖKMINGAI!")

    def discretize_state(self, continuous_state):
        """Diskretizuoja būsenas"""
        pos, vel, ang, ang_vel = continuous_state

        pos_idx = np.clip(np.digitize(pos, self.pos_bins), 0, self.n_bins)
        vel_idx = np.clip(np.digitize(vel, self.vel_bins), 0, self.n_bins)
        ang_idx = np.clip(np.digitize(ang, self.ang_bins), 0, self.n_bins)
        ang_vel_idx = np.clip(np.digitize(ang_vel, self.ang_vel_bins), 0, self.n_bins)

        return (pos_idx, vel_idx, ang_idx, ang_vel_idx)

    def _explore_environment(self, n_episodes):

        for episode in range(n_episodes):
            state = self.env.reset()[0]
            discrete_state = self.discretize_state(state)

            for step in range(500):
                action = self.env.action_space.sample()
                next_state, reward, done, _, _ = self.env.step(action)
                next_discrete_state = self.discretize_state(next_state)

                # Atlygio funkcija
                if done:
                    reward = -100
                else:
                    stability_bonus = (1 - abs(state[2]) / 0.2095) * 2
                    reward = 1 + stability_bonus

                self.transitions[(discrete_state, action)].append(next_discrete_state)
                self.rewards[(discrete_state, action)].append(reward)

                discrete_state = next_discrete_state
                state = next_state

                if done:
                    break

        print(f"✅ Surinkta {len(self.transitions)} unikalių perėjimų")
#P(s'|s,a) ir R(s,a,s') funkcijos
    def _build_transition_probabilities(self):

        for (state, action), next_states in self.transitions.items():
            next_state_counts = defaultdict(int)
            total_rewards = defaultdict(float)

            for i, next_state in enumerate(next_states):
                next_state_counts[next_state] += 1
                total_rewards[next_state] += self.rewards[(state, action)][i]

            total_transitions = len(next_states)

            if state not in self.P:
                self.P[state] = {}
                self.R[state] = {}

            self.P[state][action] = {}
            self.R[state][action] = {}

            for next_state, count in next_state_counts.items():
                prob = count / total_transitions
                avg_reward = total_rewards[next_state] / count

                self.P[state][action][next_state] = prob
                self.R[state][action][next_state] = avg_reward

    def value_iteration(self, gamma=0.95, theta=1e-4, max_iterations=1000):
        """VERTĖS ITERACIJOS ALGORITMAS"""
        print("🔄 VERTĖS ITERACIJA:")

        # Inicializacija
        for state in self.P.keys():
            self.V[state] = 0.0

        for iteration in range(max_iterations):
            delta = 0
            old_V = self.V.copy()

            for state in self.P.keys():
                if self._is_terminal(state):
                    continue

                # Bellman optimality equation
                action_values = []
                for action in range(self.n_actions):
                    if action in self.P[state]:
                        action_value = 0
                        for next_state, prob in self.P[state][action].items():
                            reward = self.R[state][action][next_state]
                            action_value += prob * (reward + gamma * old_V.get(next_state, 0))
                        action_values.append(action_value)
                    else:
                        action_values.append(-1000)

                if action_values:
                    self.V[state] = max(action_values)
                    self.policy[state] = np.argmax(action_values)
                    delta = max(delta, abs(old_V[state] - self.V[state]))

            if delta < theta:
                print(f"✅ Konvergavo po {iteration + 1} iteracijų!")
                break

        print(f"📊 Rasta strategija {len(self.policy)} būsenoms")
        return self.V, self.policy

    def _is_terminal(self, state):
        """Tikrina ar būsena yra baigtine"""
        pos_idx, vel_idx, ang_idx, ang_vel_idx = state

        if pos_idx == 0:
            pos_val = -2.4
        elif pos_idx >= len(self.pos_bins):
            pos_val = 2.4
        else:
            pos_val = self.pos_bins[pos_idx - 1]

        if ang_idx == 0:
            ang_val = -0.2095
        elif ang_idx >= len(self.ang_bins):
            ang_val = 0.2095
        else:
            ang_val = self.ang_bins[ang_idx - 1]

        return abs(pos_val) > 2.4 or abs(ang_val) > 0.2095

    def get_action(self, continuous_state):
        """Gauna optimalų veiksmą"""
        discrete_state = self.discretize_state(continuous_state)
        return self.policy.get(discrete_state, 0)

    def test_policy(self, n_episodes=5):
        """Testuoja išmoktą strategiją"""
        print("🎯 STRATEGIJOS TESTAVIMAS:")
        scores = []

        for episode in range(n_episodes):
            state = self.env.reset()[0]
            score = 0

            for step in range(500):
                action = self.get_action(state)
                state, reward, done, _, _ = self.env.step(action)
                score += 1
                if done:
                    break

            scores.append(score)
            print(f"   Epizodas {episode + 1}: {score} žingsnių")

        avg_score = np.mean(scores)
        print(f"📊 Vidutinis rezultatas: {avg_score:.1f} žingsnių")
        return avg_score

    def visualize_value_function(self):
        """Vizualizuoja V* ir π*"""
        print("📈 VIZUALIZACIJA:")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Vertės funkcijos grafikas
        pos_range = np.linspace(-2.4, 2.4, 20)
        ang_range = np.linspace(-0.2, 0.2, 20)
        V_grid = np.zeros((20, 20))
        policy_grid = np.zeros((20, 20))

        for i, pos in enumerate(pos_range):
            for j, ang in enumerate(ang_range):
                test_state = self.discretize_state([pos, 0, ang, 0])
                V_grid[j, i] = self.V.get(test_state, 0)
                policy_grid[j, i] = self.policy.get(test_state, 0)

        # V* grafikas
        im1 = ax1.imshow(V_grid, extent=[-2.4, 2.4, -0.2, 0.2],
                         origin='lower', cmap='viridis', aspect='auto')
        ax1.set_title('Vertės funkcija V*(s)')
        ax1.set_xlabel('Pozicija')
        ax1.set_ylabel('Kampas (rad)')
        plt.colorbar(im1, ax=ax1)

        # π* grafikas
        im2 = ax2.imshow(policy_grid, extent=[-2.4, 2.4, -0.2, 0.2],
                         origin='lower', cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Optimali strategija π*(s)\n(0=Kairėn, 1=Dešinėn)')
        ax2.set_xlabel('Pozicija')
        ax2.set_ylabel('Kampas (rad)')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.show()

    def demo_optimal_policy(self, n_episodes=2):
        """Demonstracija su vizualiu"""
        print("🎮 DEMONSTRACIJA:")
        demo_env = gym.make('CartPole-v1', render_mode='human')

        try:
            for episode in range(n_episodes):
                state = demo_env.reset()[0]
                for step in range(500):
                    demo_env.render()
                    action = self.get_action(state)
                    state, reward, done, _, _ = demo_env.step(action)

                    import time
                    time.sleep(0.05)

                    if done:
                        print(f"Epizodas baigtas po {step + 1} žingsnių")
                        break
        finally:
            demo_env.close()

    def close(self):
        self.env.close()


# VYKDYMAS
if __name__ == "__main__":
    # 1. Sukurti MDP ir rasti π*
    mdp = CartPoleMDP(n_bins=6, n_episodes=2000)
    V_star, policy_star = mdp.value_iteration(gamma=0.95)

    # 2. Vizualizuoti rezultatus
    mdp.visualize_value_function()

    # 3. Testuoti strategiją
    mdp.test_policy(n_episodes=5)

    # 4. Demonstracija
    input("Spauskite ENTER demonstracijai...")
    mdp.demo_optimal_policy()

    mdp.close()