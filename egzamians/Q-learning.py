import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def train_q_agent_with_tracking():

    print("Treniruojamas Q-Learning agentas...")
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
    epsilon_decay = 0.9995  # Lėtesnis decay
    epsilon_min = 0.05

    training_scores = []

    for episode in range(1500):  # Daugiau epizodų
        state = env.reset()[0]
        steps = 0

        # Diskretizuoti pradinę būseną
        s_p = np.clip(np.digitize(state[0], pos_space), 0, n_bins)
        s_v = np.clip(np.digitize(state[1], vel_space), 0, n_bins)
        s_a = np.clip(np.digitize(state[2], ang_space), 0, n_bins)
        s_av = np.clip(np.digitize(state[3], ang_vel_space), 0, n_bins)

        done = False

        while not done and steps < 500:
            # Epsilon-greedy su pagerintais parametrais
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
            print(f"   Epizodas {episode}, Vidutinis rezultatas: {avg_score:.1f}, Epsilon: {epsilon:.3f}")

    env.close()
    print("✅ Q-Learning agentas ištreniruotas!")
    return q_table, (pos_space, vel_space, ang_space, ang_vel_space), training_scores


def test_q_agent(q_table, spaces, episodes=10):
    """Test the trained Q-learning agent"""
    env = gym.make('CartPole-v1', render_mode='human')
    pos_space, vel_space, ang_space, ang_vel_space = spaces
    n_bins = len(pos_space)

    for episode in range(episodes):
        state = env.reset()[0]
        steps = 0
        done = False

        print(f"Epizodas {episode + 1}")

        while not done and steps < 500:
            # Discretize state
            s_p = np.clip(np.digitize(state[0], pos_space), 0, n_bins)
            s_v = np.clip(np.digitize(state[1], vel_space), 0, n_bins)
            s_a = np.clip(np.digitize(state[2], ang_space), 0, n_bins)
            s_av = np.clip(np.digitize(state[3], ang_vel_space), 0, n_bins)

            # Get action from Q-table
            action = np.argmax(q_table[s_p, s_v, s_a, s_av, :])

            state, reward, done, _, _ = env.step(action)
            steps += 1

        print(f"   žingsniai: {steps}")

    env.close()


def plot_training_progress(training_scores):
    """Plot training progress"""
    plt.figure(figsize=(10, 6))
    plt.plot(training_scores, alpha=0.6)

    # Moving average
    window_size = 100
    if len(training_scores) >= window_size:
        moving_avg = []
        for i in range(window_size - 1, len(training_scores)):
            moving_avg.append(np.mean(training_scores[i - window_size + 1:i + 1]))
        plt.plot(range(window_size - 1, len(training_scores)), moving_avg,
                 color='red', linewidth=2, label=f'Moving Average ({window_size})')

    plt.title('Q-Learning Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    q_table, spaces, scores = train_q_agent_with_tracking()
    plot_training_progress(scores)

    print("\nTesting trained agent...")
    test_q_agent(q_table, spaces)