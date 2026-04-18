"""
================================================
Q-Learning Code Demo
Applied Deep Learning for Civil Engineers
Lecture 12: Reinforcement Learning and Embodied AI
Dr. Jiaji Wang | The University of Hong Kong
================================================

This demo covers THREE progressive implementations:
  Part 1 – Tabular Q-Learning on Grid World (matches Slides 15–17)
  Part 2 – Q-Table Visualisation                (matches Slide 21)
  Part 3 – Deep Q-Network (DQN) with Experience Replay
                                                (matches Slides 22–31)

Civil Engineering Context:
  The robot inspector navigates a simplified
  construction floor-plan to reach an inspection
  target while avoiding hazard zones.
"""

# ─────────────────────────────────────────────
#  Dependencies (all standard in Google Colab)
# ─────────────────────────────────────────────
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# Optional: deep learning (Part 3 only)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found – Part 3 (DQN) will be skipped.")


# ════════════════════════════════════════════
#  PART 1 – GRID WORLD ENVIRONMENT
#  (Matches Slides 15-17: "A simple MDP: Grid World")
# ════════════════════════════════════════════

class ConstructionGridWorld:
    """
    A 6×6 grid representing a construction floor-plan.

    Legend:
      'S' – Start position (robot inspector)
      'G' – Goal  (inspection target / terminal state +10)
      'H' – Hazard zone (terminal state  −10)
      '.' – Free walkable cell (reward −1 per step)

    MDP components (Slide 13):
      States  S  : (row, col) grid positions
      Actions A  : {UP, DOWN, LEFT, RIGHT}
      Reward  R  : +10 goal | −10 hazard | −1 step
      Transition : deterministic (for simplicity)
      Discount γ : configurable (default 0.95)
    """

    GRID = [
        ['S', '.', '.', 'H', '.', '.'],
        ['.', 'H', '.', '.', '.', '.'],
        ['.', '.', '.', 'H', '.', 'H'],
        ['.', '.', 'H', '.', '.', '.'],
        ['.', 'H', '.', '.', 'H', '.'],
        ['.', '.', '.', '.', '.', 'G'],
    ]

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = {0: '↑ UP', 1: '↓ DOWN', 2: '← LEFT', 3: '→ RIGHT'}

    REWARDS = {'G': 10.0, 'H': -10.0, '.': -1.0, 'S': -1.0}

    def __init__(self):
        self.nrows = len(self.GRID)
        self.ncols = len(self.GRID[0])
        self.n_states = self.nrows * self.ncols
        self.n_actions = 4
        self.start = (0, 0)
        self._find_goal()
        self.reset()

    def _find_goal(self):
        for r, row in enumerate(self.GRID):
            for c, cell in enumerate(row):
                if cell == 'G':
                    self.goal = (r, c)

    def reset(self):
        self.agent_pos = self.start
        return self._state_id(self.agent_pos)

    def _state_id(self, pos):
        """Convert (row, col) → integer state index."""
        return pos[0] * self.ncols + pos[1]

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        nr = max(0, min(self.nrows - 1, self.agent_pos[0] + dr))
        nc = max(0, min(self.ncols - 1, self.agent_pos[1] + dc))
        self.agent_pos = (nr, nc)

        cell = self.GRID[nr][nc]
        reward = self.REWARDS[cell]
        done = cell in ('G', 'H')
        next_state = self._state_id(self.agent_pos)
        return next_state, reward, done

    def render(self, q_table=None, title="Construction Site Grid World"):
        """Visualise the grid with optional Q-value policy arrows."""
        fig, ax = plt.subplots(figsize=(7, 7))
        colors = {'S': '#4A90D9', 'G': '#27AE60', 'H': '#E74C3C', '.': '#ECF0F1'}

        for r in range(self.nrows):
            for c in range(self.ncols):
                cell = self.GRID[r][c]
                color = colors[cell]
                # Highlight current agent position
                if (r, c) == self.agent_pos:
                    color = '#F39C12'
                rect = plt.Rectangle([c, self.nrows - 1 - r], 1, 1,
                                      facecolor=color, edgecolor='white', lw=2)
                ax.add_patch(rect)

                # Cell label
                label = {'S': 'START', 'G': 'GOAL',
                         'H': 'HAZARD', '.': ''}[cell]
                ax.text(c + 0.5, self.nrows - r - 0.5, label,
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color='white' if cell in ('H', 'G') else '#2C3E50')

                # Draw best-action arrow from Q-table
                if q_table is not None and cell not in ('G', 'H'):
                    state = r * self.ncols + c
                    best_a = np.argmax(q_table[state])
                    dr, dc_ = self.ACTIONS[best_a]
                    ax.annotate('', xy=(c + 0.5 + dc_ * 0.3,
                                        self.nrows - r - 0.5 - dr * 0.3),
                                xytext=(c + 0.5, self.nrows - r - 0.5),
                                arrowprops=dict(arrowstyle='->', color='#2C3E50',
                                                lw=1.5))

        ax.set_xlim(0, self.ncols)
        ax.set_ylim(0, self.nrows)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

        # Legend
        patches = [mpatches.Patch(color=v, label=k)
                   for k, v in colors.items() if k != 'S']
        patches.insert(0, mpatches.Patch(color='#F39C12', label='Agent'))
        ax.legend(handles=patches, loc='upper right', fontsize=8,
                  bbox_to_anchor=(1.18, 1.0))
        plt.tight_layout()
        plt.savefig(r'D:\Q-Learning\grid_world.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure saved → grid_world.png")


# ════════════════════════════════════════════
#  PART 2 – TABULAR Q-LEARNING
#  (Matches Slides 19–21: Bellman Equation + Value Iteration)
# ════════════════════════════════════════════

class TabularQLearning:
    """
    Classic tabular Q-Learning.

    Bellman update (Slide 20):
      Q(s,a) ← Q(s,a) + α · [r + γ·max_a' Q(s',a') − Q(s,a)]

    Exploration: ε-greedy (explore with probability ε,
                            exploit with probability 1−ε)
    """

    def __init__(self, env, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha        # Learning rate
        self.gamma = gamma        # Discount factor (Slide 13)
        self.epsilon = epsilon    # Initial exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: shape (n_states, n_actions) — initialised to zero
        self.Q = np.zeros((env.n_states, env.n_actions))

    def choose_action(self, state):
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.env.n_actions - 1)  # explore
        return int(np.argmax(self.Q[state]))                   # exploit

    def update(self, state, action, reward, next_state, done):
        """
        Bellman equation update (Slide 20).
        When episode ends (done=True) there is no future reward.
        """
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def train(self, n_episodes=1000, max_steps=200, verbose=True):
        rewards_history = []
        steps_history = []

        for ep in range(n_episodes):
            state = self.env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            steps_history.append(step + 1)

            if verbose and (ep + 1) % 200 == 0:
                avg_r = np.mean(rewards_history[-200:])
                print(f"  Episode {ep+1:>5}/{n_episodes}  "
                      f"Avg Reward (last 200): {avg_r:+6.1f}  "
                      f"ε={self.epsilon:.3f}")

        return rewards_history, steps_history

    def plot_training(self, rewards, window=50):
        """Plot smoothed reward curve (analogous to learning curves in Slide 31)."""
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(rewards, alpha=0.3, color='steelblue', label='Raw')
        axes[0].plot(range(window-1, len(rewards)),
                     smoothed, color='steelblue', lw=2, label=f'MA-{window}')
        axes[0].axhline(0, color='gray', linestyle='--', lw=0.8)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Q-Learning Training Curve')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Q-value heatmap for action 3 (RIGHT) as example
        q_map = self.Q[:, 3].reshape(self.env.nrows, self.env.ncols)
        im = axes[1].imshow(q_map, cmap='RdYlGn', aspect='auto')
        plt.colorbar(im, ax=axes[1], label='Q(s, RIGHT)')
        axes[1].set_title('Q-Value Heatmap  [Action: → RIGHT]')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Row')

        plt.tight_layout()
        plt.savefig(r'D:\Q-Learning\q_learning_training.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure saved → q_learning_training.png")

    def print_q_table(self):
        """Print Q-table in readable format."""
        print("\n── Learned Q-Table (rows=states, cols=actions) ──")
        header = f"{'State':>6}  " + "  ".join(
            f"{n:>10}" for n in self.env.ACTION_NAMES.values())
        print(header)
        print("─" * (len(header) + 2))
        for s in range(self.env.n_states):
            r, c = divmod(s, self.env.ncols)
            cell = self.env.GRID[r][c]
            row_str = f"({r},{c}){cell:>2}  " + "  ".join(
                f"{self.Q[s, a]:>10.3f}" for a in range(self.env.n_actions))
            best = int(np.argmax(self.Q[s]))
            if cell not in ('G', 'H'):
                row_str += f"  ← best: {self.env.ACTION_NAMES[best]}"
            print(row_str)


# ════════════════════════════════════════════
#  PART 3 – DEEP Q-NETWORK (DQN)
#  (Matches Slides 22–31: Neural Network Q-function
#   + Experience Replay + Target Network)
# ════════════════════════════════════════════

if TORCH_AVAILABLE:

    class QNetwork(nn.Module):
        """
        Neural network that approximates Q(s, a; θ).

        Architecture (Slide 26–29):
          Input  : state (one-hot encoded, size = n_states)
          Hidden : two fully-connected layers with ReLU
          Output : Q-values for each action (size = n_actions)

        This is the "function approximator" described in Slides 21-22.
        For Atari, the input would be pixel frames through Conv layers.
        Here we use a flat state representation for the Grid World.
        """

        def __init__(self, n_states, n_actions, hidden_size=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_states, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions)
            )

        def forward(self, x):
            return self.net(x)


    class ReplayBuffer:
        """
        Experience Replay buffer (Slide 30).

        Stores transitions (s, a, r, s', done).
        Training samples random mini-batches to break temporal
        correlation between consecutive samples.

        Quote from Slide 30:
          "Samples are correlated => inefficient learning
           Address using experience replay"
        """

        def __init__(self, capacity=10_000):
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            return (np.array(states), np.array(actions),
                    np.array(rewards, dtype=np.float32),
                    np.array(next_states),
                    np.array(dones, dtype=np.float32))

        def __len__(self):
            return len(self.buffer)


    class DQNAgent:
        """
        Deep Q-Network agent with:
          • Online Q-network  (updated every step)
          • Target Q-network  (frozen, copied every C steps)
          • Experience replay (random mini-batch training)
          • ε-greedy exploration

        This implements Algorithm 1 from the DQN paper
        (Mnih et al. 2015) referenced in Slide 31.
        """

        def __init__(self, env, lr=1e-3, gamma=0.95,
                     epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.997,
                     batch_size=64, target_update_freq=100,
                     buffer_capacity=10_000, hidden_size=64):

            self.env = env
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq

            # One-hot encoding dimension = n_states
            self.n_states = env.n_states
            self.n_actions = env.n_actions

            # ── Networks ──────────────────────────────────────
            self.q_net = QNetwork(self.n_states, self.n_actions, hidden_size)
            self.target_net = QNetwork(self.n_states, self.n_actions, hidden_size)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_net.eval()  # Target network is not trained directly

            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()  # Corresponds to L(θ) in Slide 23

            # ── Replay Buffer ──────────────────────────────────
            self.memory = ReplayBuffer(buffer_capacity)
            self.steps_done = 0

        def _encode_state(self, state_id):
            """One-hot encode integer state → float tensor."""
            vec = np.zeros(self.n_states, dtype=np.float32)
            vec[state_id] = 1.0
            return vec

        def choose_action(self, state):
            if random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)
            with torch.no_grad():
                s = torch.FloatTensor(self._encode_state(state)).unsqueeze(0)
                q_vals = self.q_net(s)
                return int(q_vals.argmax().item())

        def _update_weights(self):
            """
            Mini-batch gradient descent on Bellman loss.

            Loss (Slide 23):
              L(θ) = E[(y − Q(s,a;θ))²]
              where y = r + γ·max_a' Q(s',a'; θ⁻)   ← target network θ⁻
            """
            if len(self.memory) < self.batch_size:
                return None

            states, actions, rewards, next_states, dones = \
                self.memory.sample(self.batch_size)

            # Encode states
            s_t = torch.FloatTensor(
                np.array([self._encode_state(s) for s in states]))
            s_next = torch.FloatTensor(
                np.array([self._encode_state(s) for s in next_states]))
            a_t = torch.LongTensor(actions)
            r_t = torch.FloatTensor(rewards)
            done_t = torch.FloatTensor(dones)

            # Current Q-values: Q(s,a;θ)
            q_current = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

            # Target Q-values: y = r + γ·max Q(s',a';θ⁻) · (1−done)
            with torch.no_grad():
                q_next_max = self.target_net(s_next).max(1)[0]
                y = r_t + self.gamma * q_next_max * (1 - done_t)

            # Compute and backpropagate loss
            loss = self.loss_fn(q_current, y)
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability (common DQN trick)
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            return loss.item()

        def train(self, n_episodes=1000, max_steps=200, verbose=True):
            rewards_history = []
            loss_history = []

            for ep in range(n_episodes):
                state = self.env.reset()
                total_reward = 0
                ep_losses = []

                for step in range(max_steps):
                    action = self.choose_action(state)
                    next_state, reward, done = self.env.step(action)

                    # Store transition in replay buffer (Slide 36)
                    self.memory.push(state, action, reward, next_state, done)

                    # Gradient update (Slide 37)
                    loss = self._update_weights()
                    if loss is not None:
                        ep_losses.append(loss)

                    state = next_state
                    total_reward += reward
                    self.steps_done += 1

                    # Copy Q-network → Target network every C steps
                    if self.steps_done % self.target_update_freq == 0:
                        self.target_net.load_state_dict(self.q_net.state_dict())

                    if done:
                        break

                # Decay ε
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon * self.epsilon_decay)
                rewards_history.append(total_reward)
                if ep_losses:
                    loss_history.append(np.mean(ep_losses))

                if verbose and (ep + 1) % 200 == 0:
                    avg_r = np.mean(rewards_history[-200:])
                    avg_l = np.mean(loss_history[-200:]) if loss_history else 0
                    print(f"  Episode {ep+1:>5}/{n_episodes}  "
                          f"Avg Reward: {avg_r:+6.1f}  "
                          f"Loss: {avg_l:.4f}  "
                          f"ε={self.epsilon:.3f}")

            return rewards_history, loss_history

        def plot_training(self, rewards, losses, window=50):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Reward curve
            smoothed_r = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(rewards, alpha=0.3, color='royalblue')
            axes[0].plot(range(window-1, len(rewards)), smoothed_r,
                         color='royalblue', lw=2)
            axes[0].axhline(0, color='gray', linestyle='--', lw=0.8)
            axes[0].set_title('DQN Training – Reward')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Total Reward')
            axes[0].grid(alpha=0.3)

            # Loss curve
            if losses:
                smoothed_l = np.convolve(losses, np.ones(min(window, len(losses)))
                                         / min(window, len(losses)), mode='valid')
                axes[1].plot(losses, alpha=0.3, color='tomato')
                axes[1].plot(range(len(smoothed_l)), smoothed_l,
                             color='tomato', lw=2)
                axes[1].set_title('DQN Training – Bellman Loss L(θ)')
                axes[1].set_xlabel('Episode')
                axes[1].set_ylabel('MSE Loss')
                axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(r'D:\Q-Learning\dqn_training.png',
                        dpi=150, bbox_inches='tight')
            plt.show()
            print("Figure saved → dqn_training.png")

        def get_q_table_from_network(self):
            """Extract Q-values for all states for visualisation."""
            q_table = np.zeros((self.env.n_states, self.env.n_actions))
            with torch.no_grad():
                for s in range(self.env.n_states):
                    enc = torch.FloatTensor(self._encode_state(s)).unsqueeze(0)
                    q_table[s] = self.q_net(enc).numpy()
            return q_table


# ════════════════════════════════════════════
#  MAIN – RUN ALL THREE PARTS
# ════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Q-Learning Demo  |  Applied DL for Civil Engineers")
    print("  Lecture 12: Reinforcement Learning & Embodied AI")
    print("  Dr. Jiaji Wang  |  The University of Hong Kong")
    print("=" * 60)

    env = ConstructionGridWorld()

    # ── Show the environment ──────────────────────────────────
    print("\n[Environment] Construction Site Grid World")
    env.render(title="Construction Site Grid World – Initial State")

    # ─────────────────────────────────────────────────────────
    # PART 2 : Tabular Q-Learning
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("PART 2 – Tabular Q-Learning  (Slides 19–21)")
    print("─" * 60)
    print("Hyperparameters:")
    print("  α (learning rate)  = 0.1")
    print("  γ (discount)       = 0.95")
    print("  ε start            = 1.0  → 0.05  (ε-greedy)")
    print("  Episodes           = 1000")

    agent_tab = TabularQLearning(env, alpha=0.1, gamma=0.95,
                                 epsilon=1.0, epsilon_min=0.05,
                                 epsilon_decay=0.995)
    rewards_tab, steps_tab = agent_tab.train(n_episodes=1000, verbose=True)
    agent_tab.print_q_table()
    agent_tab.plot_training(rewards_tab)
    env.render(q_table=agent_tab.Q,
               title="Learned Policy – Tabular Q-Learning\n(arrows = greedy action)")

    # ─────────────────────────────────────────────────────────
    # PART 3 : Deep Q-Network (DQN)
    # ─────────────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        print("\n" + "─" * 60)
        print("PART 3 – Deep Q-Network (DQN)  (Slides 22–31)")
        print("─" * 60)
        print("Architecture: Linear(36→64) → ReLU → Linear(64→64)")
        print("              → ReLU → Linear(64→4)")
        print("Hyperparameters:")
        print("  lr              = 1e-3  (Adam)")
        print("  γ (discount)    = 0.95")
        print("  Replay buffer   = 10,000 transitions")
        print("  Batch size      = 64")
        print("  Target update   = every 100 steps")
        print("  Episodes        = 1000")

        env.reset()
        dqn_agent = DQNAgent(env, lr=1e-3, gamma=0.95,
                             epsilon=1.0, epsilon_min=0.05,
                             epsilon_decay=0.997, batch_size=64,
                             target_update_freq=100)
        rewards_dqn, losses_dqn = dqn_agent.train(n_episodes=1000, verbose=True)
        dqn_agent.plot_training(rewards_dqn, losses_dqn)

        # Visualise DQN policy on the grid
        q_from_net = dqn_agent.get_q_table_from_network()
        env.render(q_table=q_from_net,
                   title="Learned Policy – Deep Q-Network (DQN)\n(arrows = greedy action)")
    else:
        print("\nPART 3 skipped – install PyTorch to enable DQN.")

    print("\n✅  All outputs saved to D:\\Q-Learning\\")


if __name__ == "__main__":
    main()
