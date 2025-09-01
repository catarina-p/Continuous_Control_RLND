import numpy as np
import psutil
import torch

from collections import deque
from unityagents import UnityEnvironment

from agent_p2 import Agent

import matplotlib.pyplot as plt


def cleanup_zombie_processes():
    for proc in psutil.process_iter(['pid', 'ppid', 'name', 'status']):
        if proc.info['name'] == 'Reacher_One_Lin' and proc.info['status'] == 'zombie':
            print(f"Zombie process detected: {proc.info}")
            parent = psutil.Process(proc.info['ppid'])
            print(f"Terminating parent process: {parent}")
            parent.terminate()

def ddpg(agent, env, n_episodes=500, goal=30.0, train_mode=True, episode_window=100, print_every=100):
    """ Deep Deterministic Policy  Gradient
        Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    mean_scores = []                               # list of mean scores from each episode
    min_scores = []                                # list of lowest scores from each episode
    max_scores = []                                # list of highest scores from each episode
    best_score = -np.inf
    scores_window = deque(maxlen=episode_window)  # mean scores from most recent episodes
    moving_avgs = []                               # list of moving averages
    brain_name = env.brain_names[0]
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(len(env_info.agents))                # initialize the score (for each agent)
        agent.reset()
        step_num = 0
    
        while True:
            actions = agent.act(states, noise=True)             # select an action
            env_info = env.step(actions)[brain_name]            # send the action to the environment
            next_states = env_info.vector_observations          # get the next state
            rewards = env_info.rewards                          # get the reward
            dones = env_info.local_done  
    
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, step_num)
    
            states = next_states
            score += rewards
            step_num += 1
    
            if np.any(dones):                                  # exit loop if episode finished
                break
    
        min_scores.append(np.min(scores))             # save lowest score for a single agent
        max_scores.append(np.max(scores))             # save highest score for a single agent        
        mean_scores.append(np.mean(scores))           # save mean score for the episode
        scores_window.append(mean_scores[-1])         # save mean score to window
        moving_avgs.append(np.mean(scores_window))    # save moving average
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, moving_avgs[i_episode-1]))
        if train_mode and i_episode % print_every == 0:
            torch.save(agent.actor.state_dict(), 'weights\\actor\checkpoint_'+str(i_episode)+'.pth')
            torch.save(agent.critic.state_dict(), 'weights\\critic\checkpoint_'+str(i_episode)+'.pth')
                  
        if moving_avgs[-1] >= goal and i_episode >= episode_window:
            print('\nEnvironment SOLVED in {} episodes!\tMoving Average ={:.1f} over last {} episodes'.format(\
                                    i_episode-episode_window, moving_avgs[-1], episode_window))            
            if train_mode:
                torch.save(agent.actor_local.state_dict(), 'weights\\actor\\trained_weights.pth')
                torch.save(agent.critic_local.state_dict(), 'weights\critic\\trained_weights.pth')  
            break
    goal_vec = np.ones(i_episode)*goal
    return mean_scores, moving_avgs, goal_vec

def plotData(mean_scores, moving_avgs, goal_vec):

    fig = plt.figure(figsize=(12, 10))
    plt.plot(np.arange(len(mean_scores)), mean_scores, label = 'Mean Score')
    plt.plot(moving_avgs, 'r-', label = 'Average over the last 100 episodes',linewidth = 2)
    plt.plot(goal_vec, 'k--',label = 'Goal', linewidth = 2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()

def main():

    env = UnityEnvironment(
        file_name='Reacher.x86_64',
        no_graphics=True
    )

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    mean_scores, moving_avgs, goal_vec= ddpg(agent, env, n_episodes=500, goal=30.0, train_mode=True, episode_window=100, print_every=100)

    env.close()

    # Check for zombie processes
    cleanup_zombie_processes()

    plotData(mean_scores=mean_scores, moving_avgs=moving_avgs, goal_vec=goal_vec)


    

if __name__ == "__main__":
    main()
