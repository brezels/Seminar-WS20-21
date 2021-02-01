# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:20:48 2020
RL network

Using:
Tensorflow: 1.15
keras: 2.3.1
@author: lc226
"""
import os
import numpy as np
import random
#import pandas as pd
#import matplotlib.pyplot as plt
from collections import deque

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
#import keras.backend as K

class DQ_Network:
    #
    def __init__(self,
                actions,
                features,
                learning_rate = 0.01,
                reward_decay = 0.9,
                epsilon = 0.9,
                replace_target_iter = 100,
                memory_size = 500,
                batch_size = 32,
                epsilon_increment = None):
        self.n_actions = actions #
        self.n_features = features
        self.lr = learning_rate #学习率
        self.gamma = reward_decay #Reward decay value #远见值
        self.epsilon_max = epsilon #Probability of random choice 
        self.replace_target_iter = replace_target_iter #Number of steps required to replace the target network
        self.batch_size = batch_size   #How much taken from memory each time when updated
        self.epsilon_increment = epsilon_increment # not for now,its reserve for DDQN


        #Whether to open the exploration mode, and gradually reduce the number of explorations
        if self.epsilon_increment is not None:
            self.epsilon = 0
        else:
            self.epsilon = self.epsilon_max
          
        #total learning step(determine whether to replace the target_net parameter)
        self.learning_step = 0
        
        #initialize memory
        self.memory = deque(maxlen=memory_size)#[s,a,r,s_next]
        #build network
        self.eval_model = self.build_net()
        self.target_model = self.build_net()
        self.update_target_model()          
        if os.path.exists('DQN.h5'):
            self.eval_model.load_weights('DQN.h5')
        #Configure the learning process for two networks
        self.eval_model.compile(loss='mse', optimizer=Adam(self.lr))
        self.target_model.compile(loss='mse', optimizer=Adam(self.lr)) #可能用不到，有备无患,如果出现不收敛的情况，优先检查这里
            
    def build_net(self): #会测试这样是不是全连接
        #define network parameter
        #
        inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(self.n_actions)(x)

        model = Model(inputs=inputs, outputs=x)

        return model

    def update_target_model(self):
        #Update the parameters in the target network
        self.target_model.set_weights(self.eval_model.get_weights())


    def Store_experience(self,s,a,r,s_next):        
        experience = (s,a,r,s_next)
        self.memory.append(experience)
        
    #选择行为，等模型搭建好，这里需要进一步更改
    def choose_action(self,state):
        state = np.array(state)
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_model.predict(state)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

     
#     """学习过程
#        Returns:
#            X: states
#            y: [Q_value1, Q_value2]
#    """
    def learn(self):
        # check to replace target parameters
        if self.learning_step % self.replace_target_iter == 0:
            self.update_target_model()
            print('替换target网络')
        # sample batch memory from all memory
        #batch_memory = np.random.choice(self.memory, self.batch_size) #没有考虑最初的记忆提取，需要在思考一下
        batch_memory =random.sample(self.memory, self.batch_size) #

        states = np.array([batch[0] for batch in batch_memory])
        next_states = np.array([batch[3] for batch in batch_memory])

        q_eval  = self.eval_model.predict(states)
        q_next = self.target_model.predict(next_states)
        for i, (_, action, reward, _) in enumerate(batch_memory):
            target = reward
            target += self.gamma * np.amax(q_next[i])
            q_eval[i][action] = target
        self.eval_model.fit(states, q_eval, epochs=1)
#        self.cost_memory4plot.append(cost)#Add memory
        
        #By increasing epsilon, gradually reduce the randomness of behavior
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        self.learning_step += 1
        
        loss = 1
        #loss = self.model.train_on_batch(states, q_eval)
        return loss
   
    def save_weights(self):
         self.eval_model.save_weights('DDQN.h5')
        

    
   

#if __name__ == '__main__':
#    DQN = DQ_Network(3,4)
    
       
       
#     def plot_cost(self):
#        plt.plot(np.arrange(len(self.learning_step)),self.learning_step)#没写对，还要重写
#        plt.ylabe('Cost')
#        plt.xlabe('training steps')
#        plt.show()

       
       