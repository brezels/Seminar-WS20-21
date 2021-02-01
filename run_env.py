import gym
import numpy as np
import gym_anytrading
from gym_anytrading.envs import StocksEnv, Actions, Positions,TradingEnv 
from gym_anytrading.datasets import STOCKS_GOOGL
from DQ_network import DQ_Network
import matplotlib.pyplot as plt

def dqn_test():
    env = gym.make('stocks-v0', frame_bound=(10, 2335), window_size=10)
    rList = []              # rewards for each episode
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQ_Network(action_size,state_size,memory_size = 500)
    EPISODES = 50
    total_steps = 0
    for e in range(EPISODES):
        s = env.reset()
        s = s[:,1]
        rAll = 0 
        for time in range(3000):
            # env.render()
            action = dqn.choose_action(s)
            s_, r, done, info = env.step(action)

            s_ = s_[:,1]
            dqn.Store_experience(s, action, r, s_)

            if total_steps > 1000 and total_steps % 20 == 0:
                dqn.learn()     
        
            rAll += r #cumulate the rewards we got 
            s = s_ # move to the next state
            total_steps += 1
            if done:
                rList.append(rAll)
                print(info)
                break
    return rList


if __name__ == '__main__':
    rList = dqn_test()
    plt.figure()
    plt.plot(rList)
    plt.show()

    
       