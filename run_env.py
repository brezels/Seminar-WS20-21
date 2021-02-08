import gym
import numpy as np
import gym_anytrading
from gym_anytrading.envs import StocksEnv, Actions, Positions,TradingEnv 
from gym_anytrading.datasets import STOCKS_GOOGL
from DQ_network import DQ_Network
from collections import deque
import matplotlib.pyplot as plt

def dqn_train():
    env = gym.make('stocks-v0', frame_bound=(10, 2335), window_size=10)
    rList = []              # rewards for each episode
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQ_Network(action_size,state_size,memory_size = 500)
    EPISODES = 500
    total_steps = 0
    trigger = True
    average_profit = 0
    memory = deque(maxlen=50)
    # for e in range(EPISODES):
    while trigger:
        s = env.reset()
        s = s[:,1]
        rAll = 0 
        for time in range(3000):
            # env.render()
            action = dqn.choose_action(s)
            s_, r, done, info = env.step(action)

            s_ = s_[:,1] #choose the usefull state 
            dqn.Store_experience(s, action, r, s_)

            if total_steps > 500 and total_steps % 20 == 0:
                dqn.learn()     # start learning
        
            rAll += r #cumulate the rewards we got 
            s = s_ # move to the next state
            total_steps += 1
            if done:
                rList.append(rAll)
                print(info)
                profit = info['total_profit']
                memory.append(profit)
                break
            
        if total_steps > 300000: #大概100轮的平均收益率
            for i in memory:
                average_profit += i
            average_profit = average_profit /50
            if average_profit > 1.3:
                trigger = False
                print(total_steps)
            else:
                average_profit = 0
    dqn.save_weights()
    return rList


def dqn_test():
    print("start test")
    env = gym.make('stocks-v0', frame_bound=(10, 2335), window_size=10)
    rList = []              # rewards for each episode
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQ_Network(action_size,state_size,memory_size = 500)
    EPISODES = 10
    total_steps = 0
    sum_profit = 0
    for e in range(EPISODES):
        s = env.reset()
        s = s[:,1]
        rAll = 0 
        for time in range(3000):
            # env.render()
            action = dqn.choose_action(s)
            s_, r, done, info = env.step(action)

            s_ = s_[:,1]

            rAll += r #cumulate the rewards we got 
            s = s_ # move to the next state
            total_steps += 1
            if done:
                print(info)
                sum_profit += info['total_profit']
                break
    sum_profit = sum_profit / EPISODES
    print("average profit :",sum_profit)
    return rList



if __name__ == '__main__':
    rList_train = dqn_train()
    rList_test = dqn_test()
    plt.figure()
    plt.plot(rList_train)
    plt.show()

    
       