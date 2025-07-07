Traditional policy gradient methods aim to directly maximize the expected return of a policy. However, when the update step is too large, the policy may shift too far, causing performance degradation. Methods like TRPO, similar to PPO, constrain policy updates through KL-divergence to achieve trust-region updates. The downside is that TRPO is algorithmically complex and computationally expensive. PPO was designed to provide a simple and efficient objective function that achieves the conservative update effect similar to TRPO.

The core idea of PPO is to limit the magnitude of policy updates to prevent the new policy from diverging too far from the old one. It introduces a clipping mechanism to ensure that the updates remain within a reasonable range and performs multiple optimizations using sampled data. This clipping mechanism is PPO’s most significant distinction from TRPO. In reinforcement learning policy gradient optimization, the central challenge lies in updating the policy parameters θ in a way that consistently improves and stabilizes performance.

Training procedure: data sampling → advantage estimation → constructing clipped objective → multiple rounds of mini-batch optimization → policy parameter update → iterate

**Clipping Mechanism**

This mechanism restricts the extent of policy change to ensure stable training. It enforces the policy probability ratio to remain within [1−ε, 1+ε], preventing large shifts in policy parameters and keeping the new policy close to the old one. When the ratio
rt(θ) exceeds this range, the gradient of the objective function does not increase further, thereby preventing performance degradation due to excessive optimization. The clipping mechanism implements an implicit trust region similar to TRPO by hard-clipping the policy ratio. It also reduces training instability caused by outlier actions with extreme gradients.

```
initialize policy parameters θ and value function parameters φ
    θ_old ← θ  
    
    for iteration in range(N_iterations):
        trajectories = collect_trajectories(π_θ_old)  
        
        for t in trajectories:
            compute advantage A_t using GAE
            compute value target for critic
    
        for epoch in range(K):  
            for minibatch in trajectories:
                compute r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
                compute clipped surrogate loss:
                    L_clip = min(r_t * A_t, clip(r_t, 1 - ε, 1 + ε) * A_t)
                compute value loss: L_v = (V_φ(s_t) - V_target)^2
                compute entropy loss: L_H = -entropy[π_θ]
                
                total_loss = -L_clip + c1 * L_v + c2 * L_H
                update θ, φ using gradient descent
    
        θ_old ← θ
```

PPO adopts a dual-network architecture: the actor network (policy π_θ) and the critic network (value function V_φ). The actor takes the state as input and outputs a probability distribution over actions, while the critic estimates the value of the current state to support the advantage function used in training. In each update cycle, the current policy π_θ_old is used for full interaction with the environment. This on-policy nature enhances stability and prevents interference from outdated policies.

```
for t in trajectories:
        compute advantage A_t using GAE
        compute value target for critic
```

The advantage At measures the relative merit of an action under the current policy compared to the baseline state value. GAE (Generalized Advantage Estimation) is used for its estimation. It builds on temporal-difference learning by introducing a decay factor λ to balance the trade-off between bias and variance, enhancing training stability.

```
for epoch in range(K):  
    for minibatch in trajectories:
        compute r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        compute clipped surrogate loss:
            L_clip = min(r_t * A_t, clip(r_t, 1 - ε, 1 + ε) * A_t)
```

The key to PPO lies in constructing the clipped surrogate objective. When the policy update is too large, the clip function restricts the policy probability ratio rt within the range of 1 ± ε, ensuring that the updated policy does not stray too far from the old one. The min function controls the gradient direction, allowing updates only within a safe range.

Beyond the policy objective, the value function regression loss Lv and policy entropy LH
are also weighted to form the total loss function. The value loss ensures the estimated state value approaches the true return, while the entropy term encourages exploration. The total loss is then used to update both θ and φ via gradient descent, and at the end of each training round, the new parameters are used to refresh the old policy.

PPO is the core training algorithm in several of OpenAI’s reinforcement learning systems and InstructGPT. According to various articles, PPO was chosen because, as an efficient and stable policy optimization method, it solves the instability issues of previous approaches. In complex development projects that involve continuous decision-making environments, policies often have high-dimensional outputs, long decision sequences, and must continually improve without deviating too far from the current policy. Otherwise, performance may decline. PPO’s clipped policy update mechanism ensures that each update is not overly aggressive, allows multiple optimization passes on each data batch, significantly improves sample efficiency, and remains stable and easy to tune in large-scale parallel training scenarios.

### The first experimental code

```
# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1t2uqr1HWKC1jHc6HLeGwNYmQa-i2IASb
"""

!pip install gymnasium[classic-control] torch matplotlib --quiet

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from tqdm.notebook import trange
from matplotlib import animation
from IPython.display import HTML, display

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        x = self.shared(obs)
        return self.actor(x), self.critic(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
net = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = []
    gae = 0
    next_value = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)
        next_value = values[t]
    return adv

num_episodes = 2000
rewards_history = []

clip_eps = 0.1
entropy_coef = 0.02
value_coef = 0.5
target_kl = 0.02

for episode in trange(num_episodes):
    obs, _ = env.reset()
    done = False

    obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        logits, value = net(obs_tensor)
        probs = Categorical(logits=logits)
        action = probs.sample()
        log_prob = probs.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        obs_list.append(obs)
        act_list.append(action.item())
        logp_list.append(log_prob.item())
        rew_list.append(reward)
        val_list.append(value.item())
        done_list.append(done)

        obs = next_obs

    obs_tensor = torch.tensor(obs_list, dtype=torch.float32).to(device)
    act_tensor = torch.tensor(act_list).to(device)
    logp_tensor = torch.tensor(logp_list).to(device)
    val_tensor = torch.tensor(val_list).to(device)
    rew_tensor = torch.tensor(rew_list, dtype=torch.float32).to(device)
    done_tensor = torch.tensor(done_list, dtype=torch.float32).to(device)

    adv_tensor = torch.tensor(compute_gae(rew_list, val_list, done_list), dtype=torch.float32).to(device)
    returns_tensor = adv_tensor + val_tensor
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    for _ in range(5):
        logits, values = net(obs_tensor)
        probs = Categorical(logits=logits)
        new_logp = probs.log_prob(act_tensor)
        ratio = (new_logp - logp_tensor).exp()


        approx_kl = (logp_tensor - new_logp).mean().item()
        if approx_kl > 1.5 * target_kl:
            print(f"Early stopping at episode {episode}, KL = {approx_kl:.4f}")
            break


        surr1 = ratio * adv_tensor
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_tensor
        policy_loss = -torch.min(surr1, surr2).mean()


        value_loss = ((returns_tensor - values.squeeze()) ** 2).mean()


        entropy = probs.entropy().mean()


        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rewards_history.append(sum(rew_list))

plt.figure(figsize=(10, 5))
plt.plot(rewards_history, label="Episode reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO with Entropy Bonus & KL Early Stopping")
plt.grid(True)
plt.legend()
plt.show()

def render_agent_as_gif(net, env, max_frames=500):
    frames = []
    obs, _ = env.reset()
    done = False

    for _ in range(max_frames):
        frame = env.render()
        frames.append(frame)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = net(obs_tensor)
            probs = Categorical(logits=logits)
            action = probs.sample().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()

    fig = plt.figure(figsize=(frames[0].shape[1] / 72, frames[0].shape[0] / 72), dpi=72)
    plt.axis("off")
    im = plt.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=30)
    html = ani.to_jshtml()
    plt.close()
    display(HTML(html))

render_env = gym.make("CartPole-v1", render_mode="rgb_array")
render_agent_as_gif(net, render_env)
```

⬆️ In each episode, the agent uses the current policy and value function to execute a complete trajectory in the environment, collecting data for training.

The logical flow is: obs → net → actor & critic. The action distribution π(a) is obtained from the network, then a discrete action is randomly sampled from Categorical(logits). This action is executed via env.step(action.item()), which returns the next state, reward, and done signal. All information from the current timestep is saved into a list.

```
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = []
    gae = 0
    next_value = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)
        next_value = values[t]
    return adv
```

The delta is calculated as δₜ = rₜ + γV(sₜ₊₁) − V(sₜ), representing the one-step TD error — the gap between the current state value and the actual reward plus the next state value. The GAE formula is recursively defined as Âₜ = δₜ + γλÂₜ₊₁, which is an exponentially weighted multi-step TD error that balances bias and variance to enhance training stability. If the episode terminates, the value estimate for future timesteps should be set to zero. The function returns a sequence of advantage values adv used in the policy gradient objective.

```
for _ in range(5):
    logits, values = net(obs_tensor)
    probs = Categorical(logits=logits)
    new_logp = probs.log_prob(act_tensor)

    ratio = (new_logp - logp_tensor).exp()
```

In PPO, the objective we aim to optimize is: L(θ) = Eₜ[min(rₜ(θ)Âₜ, clip(rₜ(θ), 1−ε, 1+ε)Âₜ)]
⬆️ Here, logp_tensor is the log-probability of actions under the old policy, and new_logp is that of the current policy. The ratio is the importance sampling ratio rₜ(θ).

```
surr1 = ratio * adv_tensor
surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_tensor
policy_loss = -torch.min(surr1, surr2).mean()
```

⬆️ This is the clipped surrogate loss, which prevents aggressive policy updates.

```
entropy = probs.entropy().mean()
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

The entropy term encourages the policy to maintain randomness and exploration. If the policy becomes deterministic too early in training, it can easily fall into local optima.

probs.entropy(): Entropy of the current policy distribution. Higher entropy means more randomness.
entropy_coef: Coefficient for the entropy term (0.02 in the code).
-entropy: Higher entropy leads to a lower total loss, thus encouraging exploration.
value_loss: MSE loss for the critic.
policy_loss: Main PPO loss for the actor.

For visualization, I plotted a training reward curve where the x-axis represents the number of episodes and the y-axis shows the total reward accumulated per episode (sum(rew_list)), illustrating the performance trend of the agent. I also generated a GIF of the agent's execution process: using the trained policy to run a full episode, saving each frame and converting them into an HTML animation.

Why did I start with this beginner-level implementation? Because in Actor-Critic architectures, the separation between actor and critic is explicit, and the flow of sampling → optimization → logging is clear. The core mechanisms are also easy to observe. The environment used is the classic CartPole-v1, which has a low-dimensional state space (4D), a discrete action space (2 actions), and a simple, consistent reward structure (+1 per timestep).

However, in the current version, only one batch of data is used to update the policy per episode, and the target_kl is fixed. There's no multi-episode sampling, nor masking for handling trajectories of varying lengths.

### PPO Optimization

Building upon the initial version, which was a minimal PPO prototype, I conducted further optimizations inspired by the OpenAI paper Proximal Policy Optimization Algorithms.

In the first version, single-environment sampling was inefficient. It used a serial gym.make("CartPole-v1"), where each episode was sampled sequentially in a single environment. This single-threaded sampling bottlenecked training speed and increased sample correlation. Since PPO relies on mini-batch SGD to reduce variance, the first version applied 5 full passes on each complete trajectory without shuffling or batching.

The first version also didn't use value clipping for the critic. The value loss was simply: *value_loss = ((returns_tensor - values.squeeze()) * * 2).mean() *，This pure MSE approach lacks PPO’s typical value clipping technique, which can lead to overly aggressive critic updates, in turn destabilizing the actor.

Although an early stopping mechanism was implemented by comparing the current KL divergence to a fixed target_kl, the method was crude: if any epoch exceeded the threshold, training for the entire episode would terminate early. This often led to wasted gradients. Additionally, there was no scheduling or dynamic adjustment. Hyperparameters like entropy_coef and others were fixed and not annealed over the course of training.

To address these limitations, the second version introduced optimization strategies such as parallel environment sampling to improve efficiency and sample diversity. I used SyncVectorEnv to create 8 parallel environments, which greatly enhanced the speed of experience collection and the variety of samples. The collect_trajectory function was used to batch-sample experiences efficiently, allowing the agent to gather sufficient training data in less time and reducing temporal correlation between samples.

The primary motivation for using parallel environments was to mitigate sample autocorrelation.

> In policy gradient reinforcement learning algorithms such as PPO, the agent interacts with the environment to obtain state-action-reward sequences. When using single-threaded serial environment sampling, the collected samples are often highly temporally correlated due to the slow-changing nature of state distributions under the Markov assumption. This results in degraded sampling distribution, redundant data, and increased variance, ultimately affecting the reliability of policy estimation and the speed of training convergence. To mitigate this issue, parallel environment sampling (e.g., SyncVectorEnv or SubprocVectorEnv) is commonly adopted. By running the policy simultaneously in multiple independent environments, the trajectories of each environment evolve independently, which enhances experience diversity and reduces temporal dependency between samples. This strategy effectively breaks the sequential dependency structure of samples, approximating the i.i.d. (independent and identically distributed) assumption, thereby enabling mini-batch SGD optimizers to estimate gradients more effectively and improving the robustness and generalization of policy optimization.

`env = SyncVectorEnv([make_env() for _ in range(8)])`

With this setup, each policy update is based on experience drawn from a broader range of state distributions, aligning with the theoretical assumption of maximizing the objective function under approximately stationary policy distributions.

In the update_policy function of my second version, I introduced random shuffling and mini-batch training via SGD, which is a core update strategy in PPO:

```
idx = torch.randperm(len(obs_tensor))
for i in range(0, len(obs_tensor), minibatch_size):
    batch_idx = idx[i:i+minibatch_size]
```

The second version also incorporated Clipped Value Loss to ensure stable training of the critic. While the first version used a simple MSE loss, the second version implemented the Clipped Value Function Loss as recommended in the PPO paper. This mechanism effectively limits the magnitude of critic updates and prevents overfitting to the returns:

```
value_pred_clipped = batch_val_old + (values.squeeze() - batch_val_old).clamp(-clip_eps, clip_eps)
value_loss1 = (values.squeeze() - batch_ret).pow(2)
value_loss2 = (value_pred_clipped - batch_ret).pow(2)
value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
```

Dynamic exploration scheduling was also introduced through an entropy coefficient annealing mechanism:

```
entropy_coef = max(0.001, entropy_init * (1 - episode / num_episodes))
```

As training progresses, entropy_coef gradually decreases, guiding the policy from early-stage exploration to later-stage stability. This smooth transition from exploration to exploitation supports more efficient training.

The second version of the code also implemented reward curve smoothing using EMA and an early best policy saving mechanism. Compared to the first version, which only recorded raw reward curves, the second version also tracks the best-performing model parameters. This ensures that the training process not only follows reward trends but also maintains a deployable fallback policy:

```
if smoothed_reward > best_reward:
    best_reward = smoothed_reward
    torch.save(net.state_dict(), "best_ppo_cartpole.pth")
```

In the compute_gae function, the second version uses torch.zeros_like() for vectorized initialization and includes the bootstrapped value of the final state, making it more consistent with the GAE formulation presented in the paper.

The core idea behind Generalized Advantage Estimation (GAE) is to combine TD errors at multiple time scales using exponentially decaying weights (via λ) to obtain a more stable advantage estimate. GAE provides flexible control over the bias-variance trade-off, significantly enhancing the effectiveness and stability of training in policy gradient methods. As a result, it has been widely adopted by advanced algorithms such as PPO.

Returning to Section 3 of the paper.

The motivation for this work stems from the limitations of existing reinforcement learning methods. The introduction explicitly outlines the challenges of three major approaches: *“Q-learning (with function approximation) fails on many simple problems and is poorly understood, vanilla policy gradient methods have poor data efficiency and robustness; and trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing.”*

In particular, for policy gradient methods, the paper emphasizes a core issue: *“standard policy gradient methods perform one gradient update per data sample.”* While it may seem appealing to optimize the same objective function multiple times using the same trajectory, the paper warns that *“doing so is not well-justified, and empirically it often leads to destructively large policy updates.”*

The elegance of the clipping mechanism lies in its differentiated handling of various scenarios. The paper illustrates this clearly with graphs showing how the clipping function behaves under positive and negative advantages. When the advantage function hat{A}_t > 0, meaning the current action is better than average, and if the probability ratio r_t(theta) > 1 + epsilon, the clipping mechanism restricts it to 1 + epsilon preventing overly aggressive increases in the probability of good actions. Conversely, when hat{A}_t < 0, indicating a suboptimal action, and r_t(theta) < 1 - epsilon, the mechanism clips it to 1 - epsilon, thereby avoiding excessive penalization of poor actions.

The core philosophy behind this design is conservatism: *“With this scheme, we only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse.”* By taking the minimum between the unclipped and clipped objectives, PPO ensures that the surrogate objective is always a pessimistic lower bound of the unclipped one. This conservative estimate prevents over-optimistic updates and helps stabilize training. A figure in the paper further validates this by showing that $L^{CLIP} is indeed a lower bound of L^{CPI}and reaches its maximum around a KL divergence of 0.02.

Another key innovation of the PPO algorithm is its support for multiple optimization epochs on the same batch of data. Section 5 details the implementation: *“Each iteration, each of N (parallel) actors collect T timesteps of data. Then we construct the surrogate loss on these NT timesteps of data, and optimize it with minibatch SGD (or usually for better performance, Adam), for K epochs.”*

The specific implementation flow is as follows: first, each of the N parallel actors collects T timesteps of data, resulting in NT samples. A surrogate loss function is then constructed on this dataset, and it is optimized for K epochs using mini-batch SGD, where the batch size $M \leq NT$. The paper also highlights the choice of optimizer: *“optimize it with minibatch SGD (or usually for better performance, Adam).”* While theoretically SGD is sufficient, in practice, Adam consistently yields better performance.

The experiments in the paper validate these advantages—PPO demonstrates superior sample efficiency compared to traditional methods across a variety of benchmark tasks.

[The second version of the code](https://github.com/Constantine13th/Integration-of-practice-and-learning/blob/main/Machine%20Learning/code/ppo.py "The second version of the code")

