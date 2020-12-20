# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 20:27:09 2020

@author: Tianhao Peng
"""
import gym
import numpy as np
import gym_anytrading
from gym_anytrading.envs import StocksEnv, Actions, Positions,TradingEnv 
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt


env = gym.make('stocks-v0', frame_bound=(10, 510), window_size=10)

Q = np.zeros([1, env.action_space.n]) #create Q table

lr = .85                # alpha, if use value function approximation, we can ignore it
lambd = .90             # decay factor
num_episodes = 1000    # 迭代次数，也就是开始10000次游戏
rList = []              # rewards for each episode
dic = {}
count = 0
for i in range(num_episodes):
    s = env.reset()
    if not dic.get(str(s)):
        dic[str(s)] = count
        count += 1
        Q = np.append(Q,np.zeros([1, env.action_space.n]),axis = 0)
    rAll = 0 str(s)
    
    for j in range(3000):
        # if not dic.get(str(s)):
        #     print(s)
        #     dic[str(s)] = count
        #     count += 1
        #     Q = np.append(Q,np.zeros([1, env.action_space.n]),axis = 0)
        pre = dic.get()
        a= np.argmax(Q[pre, :] + np.random.randn(1, env.action_space.n)* (1. / (i + 1)))
        s1, r, d, info = env.step(a)
        
        if not dic.get(str(s1)):
            dic[str(s1)] = count
            count += 1
            Q = np.append(Q,np.zeros([1, env.action_space.n]),axis = 0)
        cur = dic.get(str(s1))
        Q[pre, a] = Q[pre, a] + lr * (r + lambd * np.max(Q[cur, :]) - Q[pre, a]) # renew Q table
        
        rAll += r #cumulate the rewards we got 
        s = s1 # move to the next state
        if d ==True:
            print("the " + str(i) +"th time: "+str(info))
            break
    rList.append(rAll)
    plt.figure()
    plt.plot(rList)
    plt.show()

    