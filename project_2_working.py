# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 01:08:20 2018

@author: Hari
"""

import gym
import random
import numpy as np
from random import randint
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import matplotlib.pyplot as plt
seed = 1

class Store_samples(object):
    def __init__(self):
        self.size = 50000
        self.s_t =  []
        self.s_a =  []
        self.s_r =  []
        self.s_t1 = []
        self.done = []
    
    def set_size(self, size):
        self.size = size
    
    def add_sample (self, s_t, s_a, s_r, s_t1, done):
        self.s_t.append(s_t)
        self.s_a.append(s_a)
        self.s_r.append(s_r)
        self.s_t1.append(s_t1)
        self.done.append(done)
        if (len(self.s_t) > self.size):
            self.s_t.pop(0)
            self.s_a.pop(0)
            self.s_r.pop(0)
            self.s_t1.pop(0)
            self.done.pop(0)
            
    def get_curr_size(self):
        return(len(self.s_t))
            
    def get_rand_samples (self, num_samples):
        m_size = self.get_curr_size()
        m_count = num_samples
        m_index = random.sample(range(m_size), num_samples)
        m_s_t =  [self.s_t[m_i] for m_i in m_index]
        m_s_a =  [self.s_a[m_i] for m_i in m_index]
        m_s_r =  [self.s_r[m_i] for m_i in m_index]
        m_s_t1 = [self.s_t1[m_i] for m_i in m_index]
        m_done = [self.done[m_i] for m_i in m_index]
        return (m_s_t, m_s_a, m_s_r, m_s_t1, m_done)
    

        
class Agent(object):
    def __init__(self, state_size, action_size):
        self.input_size = state_size
        self.output_size = action_size
        self.storage = Store_samples()
        self.action_model = self.create_nn_model()
        self.target_model = keras.models.clone_model(self.action_model)
        self.target_model.set_weights(self.action_model.get_weights())
        self.m_eplison_update = 400
        self.m_target_model_update = 500
        self.m_min_sample_size = 10000
        self.batch_size = 32
        self.alpha = 0.99
        self.epsilon = 1.0
        self.decay = 0.995
        
        
    def create_nn_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,8)))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(self.output_size))
        model.add(Activation('linear'))
        m_opt = Adam(lr=0.00025, decay=0.0)
        model.compile(optimizer=m_opt,
              loss='mean_squared_error')
        model.summary()
        return model
    
    def update_experience(self, m_s, m_a, m_r, m_st, m_done):
        self.storage.add_sample(m_s, m_a, m_r, m_st, m_done)
        
    def copy_action_to_target(self, m_num_steps):
        if (m_num_steps%self.m_target_model_update == 0):
            #print ("updating")
            #self.target_model.set_weights(self.action_model.get_weights())
            self.target_model = keras.models.clone_model(self.action_model)
            self.target_model.set_weights(self.action_model.get_weights())

     
    def upd_epsilon(self, num_episodes):
        self.epsilon = self.epsilon*self.decay
         
    # Hari: FIX ME : Update this to Double DQN        
    def get_action (self, state, num_episodes, test_mode):         
        #Only do random till num_episodes = m_epsilon_update
        if test_mode:
            m_s = np.array([state])
            m_res = np.argmax(self.action_model.predict(m_s)[0])
            return m_res
        else:
            if (random.random() <= self.epsilon):
                return random.randint(0, self.output_size -1)
            else:
                m_s = np.array([state])
                #print (m_s.shape, state.shape)
                m_res = np.argmax(self.action_model.predict(m_s)[0])
                #m_res2 = np.argmax(self.target_model.predict(m_s)[0])
                #print (m_res, m_res2)
                return m_res            

        
    def run_agent(self):
        m_num_samples = self.storage.get_curr_size()
        if m_num_samples < self.m_min_sample_size:
            return
        else:
            m_s, m_a, m_r, m_st, m_d = self.storage.get_rand_samples(self.batch_size)
        m_in_s  = np.array(m_s)
        m_in_st = np.array(m_st)
        
        m_Q  = self.action_model.predict(m_in_s)
        m_best_action_DDQN = self.action_model.predict(m_in_st)
        #print (m_best_action_DDQN )
        m_Q_hat = self.target_model.predict(m_in_st)
        m_y = np.zeros((self.batch_size , self.output_size))
        for i in range(self.batch_size):
            m_y[i,:] = m_Q[i]
            if (m_d[i]):
                m_y[i, m_a[i]] = m_r[i]
            else:
                m_test_res = m_r[i] + self.alpha*(m_Q_hat[i, np.argmax(m_best_action_DDQN[i])])
                m_y[i, m_a[i]] = m_test_res
        self.action_model.fit(m_in_s, m_y, batch_size=self.batch_size, epochs=1, verbose=False) 
        
        
class Solve_Lunar_Lander(object):
    def __init__(self, num_episodes = 5000):
        self.num_episodes = num_episodes
        self.agent = Agent(8, 4)
        self.num_steps = 0
        self.total_reward = 0.0
        self.m_test_rew_list = []
        self.m_train_rew_list = []
        self.test_mode = False
        
    def create_env(self):
        self.env = gym.make('LunarLander-v2')
        
    def save_to_csv(self):
        a = np.asarray(self.m_train_rew_list)
        np.savetxt("train.csv", a, delimiter=",")
        
    def save_to_test_csv(self):
        a = np.asarray(self.m_test_rew_list)
        np.savetxt("test.csv", a, delimiter=",")
        
    def plot_reward (self):
        #print ("do_nothing")
        per_data=np.genfromtxt('train.csv',delimiter=',')
        plt.plot(per_data)
        plt.xlabel ('Episodes')
        plt.ylabel ('Reward')
        plt.title('Train Reward vs Episode')
        plt.show()
        
    def plot_test_reward (self):
        #print ("do_nothing")
        per_data=np.genfromtxt('test.csv',delimiter=',')
        plt.plot(per_data)
        plt.xlabel ('Episodes')
        plt.ylabel ('Reward')
        plt.title('Test Reward vs Episode')
        plt.show()
        
    def set_episode(self, episodes):
        self.num_episodes = episodes
        
    def set_test_mode(self):
        self.test_mode = True

        
    def solve(self):
        env = gym.make('LunarLander-v2')
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)
        #print (env.observation_space.shape[0], env.action_space.n)
        for i in range(self.num_episodes):
            m_cur_state = env.reset()
            done = 0.0
            t = 0
            self.total_reward = 0.0
            self.agent.upd_epsilon(i) ## Added new
            #for t in range(199):
            while not done:
                self.num_steps += 1
                t += 1
                #env.render()
                #print(observation)
                m_cur_state = np.reshape(m_cur_state, (1,8))
                action = self.agent.get_action(m_cur_state, i, self.test_mode)
                #print (action)
                next_state, reward, done, info = env.step(action)
                self.total_reward += reward
                next_state = np.reshape(next_state, (1,8))
                #self.agent.update_experience(m_cur_state, action, reward, next_state, done)
                #self.agent.run_agent()
                if (not self.test_mode):
                    m_clipped_rew = reward/1.0
                    self.agent.update_experience(m_cur_state, action, m_clipped_rew, next_state, done)
                    self.agent.run_agent()                   
                    self.agent.copy_action_to_target(self.num_steps)
                #print(t, reward, next_state)
                m_cur_state = next_state
                if done:
                    print("Episode {} finished after {} timesteps with reward {} last reward {}".format(i, t+1, self.total_reward, reward))
            self.m_train_rew_list.append(self.total_reward)
            if (self.test_mode):
                self.m_test_rew_list.append(self.total_reward)
        
    
m_solve_LL = Solve_Lunar_Lander(750)       
m_solve_LL.solve()
m_solve_LL.save_to_csv()
m_solve_LL.plot_reward()
m_solve_LL.set_episode(100)
m_solve_LL.set_test_mode()
m_solve_LL.solve()
m_solve_LL.save_to_test_csv()
m_solve_LL.plot_test_reward()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        