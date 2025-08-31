import numpy as np
from collections import namedtuple, deque

from sumTree_p2 import SumTree

import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prioritized Replay parameters
# In the orignal paper, alpha~0.7 and beta_i~0.5
SMALL = 0.0001  #P(i)~(|TD_error|+SMALL)^\alpha
alpha = 0.7 #0.8     #P(i)~(|TD_error|+SMALL)^\alpha
beta_i = 0.5 #0.7    #w_i =(1/(N*P(i)))^\beta
beta_f = 1.
beta_update_steps = 1000

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
        # Element-wise Huber loss
        loss = F.smooth_l1_loss(Q_expected, Q_target, reduction="none")
        # Apply IS weights
        weighted_loss = (weights * loss).mean()
        return weighted_loss

    
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
