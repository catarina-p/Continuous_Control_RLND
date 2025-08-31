import numpy as np
import random
from collections import namedtuple, deque

from actor_critic_p2 import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 20       # learning timestep interval
NUM_PASSES = 10         # number of learning passes
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process

# Prioritized Replay parameters
# In the orignal paper, alpha~0.7 and beta_i~0.5
SMALL = 0.0001  #P(i)~(|TD_error|+SMALL)^\alpha
alpha = 0.7 #0.8     #P(i)~(|TD_error|+SMALL)^\alpha
beta_i = 0.5 #0.7    #w_i =(1/(N*P(i)))^\beta
beta_f = 1.
beta_update_steps = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.epsilon = EPSILON

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.replay = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every PRIMARY_UPDATE steps)
        self.prim_step = 0
        # Initialize time step (for updating every TARGET_UPDATE steps)
        self.target_step = 0

        # self.critic_optimizer.zero_grad()
        # self.actor_optimizer.zero_grad()

        # Internal batch counter
        self.counter = BATCH_SIZE
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.replay.add(state, action, reward, next_state, done)
        self.counter -= 1
        
        # Learn every UPDATE_EVERY time steps.
        self.prim_step = (self.prim_step + 1) #PRIMARY_UPDATE
        if self.prim_step == 0:
            # If enough samples are available in memory, get random subset and learn
            # if len(self.replay) > BATCH_SIZE:
            if self.counter <= 0:
                states, actions, rewards, next_states, dones, priorities, indices = self.replay.sample()
                self.learn(states, actions, rewards, next_states, dones, priorities, indices, GAMMA)

        #Update the target_network
        self.target_step = (self.target_step + 1) #TARGET_UPDATE
        if self.target_step == 0:
            # soft update
            # for target_param, local_param in zip(self.target_network.parameters(), self.primary_network.parameters()):
            #             target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

            # hard update
            with torch.no_grad():
                self.target_network.load_state_dict(self.primary_network.state_dict())
                self.target_network.eval()
                # print("target updated")
            
            # ------------------- update target network ------------------- #
            # self.soft_update(self.primary_network, self.target_network, TAU)   
        
        # Update beta for PER at the end of the episode
        if done:
            self.replay.update_beta

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.primary_network.eval()
        with torch.no_grad():
            action_values = self.primary_network(state)
        self.primary_network.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, states, actions, rewards, next_states, dones, priorities, indices, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # states, actions, rewards, next_states, dones, priorities, indices = experiences

        priority_weights = self.replay.compute_weights(np.array(priorities))

        # Prediction
        Q_expected = self.primary_network(states).gather(1, actions)

        # Compute target Q-values using Double DQN
        with torch.no_grad():
            # Compute action with max Q-value using primary network
            max_actions = self.primary_network(next_states).argmax(1, keepdim=True)
            # Compute Q-values for next states using target network
            Q_targets_next = self.target_network(next_states).gather(1, max_actions)
            # Compute target Q-values
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        self.replay.update_priorities(Q_expected, Q_targets, indices)

        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)
        loss = self.replay.PER_loss(Q_expected, Q_targets, priority_weights)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()     

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration in DDPG.
    
    Ref: Uhlenbeck & Ornstein (1930), used in Lillicrap et al. (2015).
    """

    def __init__(self, size, seed, mu=0.0):
        """
        Params
        ======
            size (int): dimension of the noise (usually same as action space)
            seed (int): random seed
            mu (float): long-running mean
            theta (float): speed of mean reversion
            sigma (float): scale of random fluctuations
        """
        self.mu = mu * np.ones(size)
        self.theta = OU_THETA
        self.sigma = OU_SIGMA
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.copy(self.mu)

    def sample(self):
        """Update internal state and return as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

# Source: https://github.com/eyalbd2/Deep_RL_Course/blob/master/Acrobot/SumTree.py
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.tree_memory = SumTree(buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
       
        self.alpha = alpha
        self.beta = beta_i
        self.step_beta = 1       

        # clipping [-1,1] is used in a 'custom loss function'
        self.p_max = 1.+SMALL #initial priority with max value in [-1,1]
        self.priorities = deque(maxlen = buffer_size) #Importance sampling weights
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = tuple((state, action, reward, next_state, done, self.p_max))
        # priority = TD_error
        self.tree_memory.add(self.p_max, e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = []
        priorities = []
        indices = []
        segment = self.tree_memory.total()/self.batch_size #total error

        for i in range(self.batch_size):
            a = segment*i
            b = segment*(i+1)
            rd = np.round(np.random.uniform(a,b),6) #should fix some problems with idx
            idx, priority, data = self.tree_memory.get(rd)
           
            experiences.append( data + (priority,) + (idx,) )

        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        for e in experiences: 
            if e is not None:
                priorities.append(e[6])
                indices.append(e[7])

        return states, actions, rewards, next_states, dones, priorities, indices

    # def PER_loss(self, input, target, weights):
    #     #Custom loss: prioritized replay introduces a bias,
    #     #             corrected with the importance-sampling weights.
    #     #input: input -- Q
    #     #       target -- r + gamma*Qhat(s', argmax_a' Q(s',a'))
    #     #       weights -- importance sampling weights
    #     #output:loss -- unbiased loss

    #     with torch.no_grad():
    #         tw = torch.tensor(weights).detach().float().to(device)

    #     loss = torch.clamp((input-target),-1,1)
    #     loss = loss**2
    #     loss = torch.sum(tw*loss)
    #     return loss
    

    def PER_loss(self, Q_expected, Q_target, weights):
        # weights: importance sampling weights, shape (batch_size,)
        td_error = Q_expected - Q_target
        # Element-wise Huber loss
        loss = F.smooth_l1_loss(Q_expected, Q_target, reduction="none")
        # Apply IS weights
        weighted_loss = (weights * loss).mean()
        return weighted_loss, td_error.detach()

    
    def update_beta(self):
        # linearly increasing from beta_i~0.5 to beta_f = 1
        self.beta = (beta_f-beta_i)*(self.step_beta-1)/(beta_update_steps-1) + beta_i

    # def compute_weights(self, priorities):
    #     #compute importance sampling weight, before the update
    #     self.priorities.append(priorities)
    #     self.p_max = np.max(self.priorities)
    #     weights = (np.sum(self.priorities)/(len(self.priorities)*priorities)) #.reshape(-1,1)**self.beta
    #     weights /= self.p_max
    #     return weights

    def compute_weights(self, priorities):
        # Compute sampling probabilities
        probs = priorities / self.tree_memory.total()
        N = len(self.tree_memory.data)  # current number of stored experiences
        # Importance-sampling weights
        weights = (N * probs) ** (-self.beta)
        weights /= weights.max()  # normalize for stability
        return weights


    def update_priorities(self, Qexpected, Qtarget, indices):
        with torch.no_grad():
                p = torch.abs(Qtarget - Qexpected)
                p = (p.cpu().numpy()+ SMALL)**alpha
                for j, idx in enumerate(indices):
                    self.tree_memory.update(idx, p[j])
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
