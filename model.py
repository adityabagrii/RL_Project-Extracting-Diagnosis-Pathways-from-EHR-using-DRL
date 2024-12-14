#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import random
import os
import copy
import gym
from gym import Env
from gym.spaces import Discrete, Box


# ### Defining some commonly used constants

# In[6]:


SEED = 42

MAX_STEPS = 10

ACTION_SPACE = ['No anemia', 'Vitamin B12/Folate deficiency anemia', 'Unspecified anemia', 'Anemia of chronic disease', 
'Iron deficiency anemia', 'Hemolytic anemia', 'Aplastic anemia', 'Inconclusive diagnosis', 'hemoglobin', 'ferritin', 'ret_count',
'segmented_neutrophils', 'tibc', 'mcv', 'serum_iron', 'rbc', 'gender', 'creatinine', 'cholestrol', 'copper', 'ethanol', 'folate', 
'glucose', 'hematocrit', 'tsat']

CLASS_DICT = {'No anemia': 0, 'Vitamin B12/Folate deficiency anemia': 1, 'Unspecified anemia': 2, 'Anemia of chronic disease': 3, 
'Iron deficiency anemia': 4, 'Hemolytic anemia': 5, 'Aplastic anemia': 6, 'Inconclusive diagnosis': 7}

FEATURE_NUM = len(ACTION_SPACE) - len(CLASS_DICT)


# ### Defining the environment

# In[7]:


# NOTE: The environment code is based from the original source code of the paper with the github link below
# https://github.com/lilly-muyama/anemia_diagnosis_pathways
class AnemiaEnv(Env):
    def __init__(self, X, Y, random=True):
        super(AnemiaEnv, self).__init__()
        self.action_space = Discrete(len(ACTION_SPACE))
        self.observation_space = Box(0, 1.5, (FEATURE_NUM,))
        self.actions = ACTION_SPACE
        self.max_steps = MAX_STEPS
        self.X = X
        self.Y = Y
        self.sample_num = len(X)
        self.idx = -1
        self.x = np.zeros((FEATURE_NUM,), dtype=np.float32)
        self.y = np.nan
        self.state = np.full((FEATURE_NUM,), -1, dtype=np.float32)
        self.num_classes = len(CLASS_DICT)
        self.episode_length = 0
        self.trajectory = []
        self.total_reward = 0
        self.random = random
        self.seed()

    def seed(self, seed=SEED):
        '''
        Defining a seed for the environment
        '''
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        A step in the environment
        '''
        if isinstance(action, np.ndarray):
            action = int(action)

        self.episode_length += 1
        reward = 0  # Default reward for each step
        
        if action < self.num_classes:  # Diagnosis action
            if action == self.y:  # Correct diagnosis
                reward += 5  # Strong positive reward for correct diagnosis
                self.total_reward += 5
                is_success = True
            else:  # Incorrect diagnosis
                reward -= 2  # Moderate penalty for incorrect diagnosis
                self.total_reward -= 2
                is_success = False
            terminated = True
            done = True
            y_actual = self.y
            y_pred = action
        elif self.actions[action] in self.trajectory:  # Repeated action
            reward -= 1  # Penalize repeated actions
            self.total_reward -= 1
            action = CLASS_DICT['Inconclusive diagnosis']  # Assign "Inconclusive diagnosis"
            terminated = True
            done = True
            y_actual = self.y
            y_pred = action
            is_success = True if y_actual == y_pred else False
        elif self.episode_length == self.max_steps:  # Timeout
            reward -= 1  # Mild penalty for running out of steps
            self.total_reward -= 1
            action = CLASS_DICT['Inconclusive diagnosis']  # Assign "Inconclusive diagnosis"
            terminated = True
            done = True
            y_actual = self.y
            y_pred = action
            is_success = True if y_actual == y_pred else False
        else:  # Query a feature
            reward += 0.1  # Small positive reward for exploring new features
            self.total_reward += 0.1
            self.state = self.get_next_state(action - self.num_classes)
            terminated = False
            done = False
            y_actual = np.nan
            y_pred = np.nan
            is_success = None

        self.trajectory.append(self.actions[action])
        info = {
            'index': self.idx,
            'episode_length': self.episode_length,
            'reward': self.total_reward,
            'y_pred': y_pred,
            'y_actual': y_actual,
            'trajectory': self.trajectory,
            'terminated': terminated,
            'is_success': is_success
        }
        return self.state, reward, done, info


    # def step(self, action):
    #     '''
    #     A step in the environment
    #     '''
    #     if isinstance(action, np.ndarray):
    #         action = int(action)

    #     self.episode_length += 1
    #     reward = 0
    #     if self.episode_length == self.max_steps: #reached maximum steos
    #         reward -=1
    #         self.total_reward -=1
    #         terminated = True
    #         done = True
    #         y_actual = self.y
    #         y_pred = CLASS_DICT['Inconclusive diagnosis']
    #         is_success = True if y_actual==y_pred else False
    #     elif action < self.num_classes: #diagnosis action
    #         if action == self.y:
    #             reward +=1
    #             self.total_reward += 1
    #             is_success = True
    #         else: 
    #             reward -= 1
    #             self.total_reward -= 1
    #             is_success = False
    #         terminated = False
    #         done = True
    #         y_actual = self.y
    #         y_pred = action
    #     elif self.actions[action] in self.trajectory: #terminate episode in case of repeated action
    #         action = CLASS_DICT['Inconclusive diagnosis']
    #         terminated = True
    #         reward -= 1
    #         self.total_reward -= 1
    #         done=True
    #         y_actual = self.y
    #         y_pred = action
    #         is_success = True if y_actual==y_pred else False
    #     else:
    #         terminated = False
    #         reward += 0.0
    #         self.total_reward += 0.0
    #         done = False
    #         self.state = self.get_next_state(action-self.num_classes)
    #         y_actual = np.nan
    #         y_pred = np.nan
    #         is_success = None
    #     self.trajectory.append(self.actions[action])
    #     info = {'index': self.idx, 'episode_length':self.episode_length, 'reward': self.total_reward, 'y_pred': y_pred, 
    #             'y_actual': y_actual, 'trajectory':self.trajectory, 'terminated':terminated, 'is_success': is_success}
    #     return self.state, reward, done, info
            
    
    def render(self):
        '''
        A representation of the current state of the environment
        '''
        print(f'STEP {self.episode_length} for index {self.idx}')
        print(f'Current state: {self.state}')
        print(f'Total reward: {self.total_reward}')
        print(f'Trajectory: {self.trajectory}')
        
    
    def reset(self, idx=None):
        '''
        Resetting the environment
        '''
        if idx is not None:
            self.idx = idx
        elif self.random:
            self.idx = random.randint(0, self.sample_num-1)
        else:
            self.idx += 1
            if self.idx == len(self.X):
                raise StopIteration()
        self.x, self.y = self.X[self.idx], self.Y[self.idx]
        self.state = np.full((FEATURE_NUM,), -1, dtype=np.float32)
        self.trajectory = []
        self.episode_length = 0
        self.total_reward = 0
        return self.state
        
    
    def get_next_state(self, feature_idx):
        '''
        Getting the next state of the environment
        '''
        self.x = self.x.reshape(-1, FEATURE_NUM)
        x_value = self.x[0, feature_idx]
        next_state = copy.deepcopy(self.state)
        next_state[feature_idx] = x_value
        return next_state


# ### Creating and Training the DQN Network

# In[8]:


data = pd.read_csv('data/train_set_basic.csv')


# In[9]:


data.head()


# In[10]:


data['label'].value_counts()


# In[11]:


data.fillna(-1, inplace=True)

X_train = data.iloc[:, 0:-1]
y_train = data.iloc[:, -1]

print(X_train.head())
print(y_train.head())

X_train, y_train = np.array(X_train), np.array(y_train)


# In[12]:


environment = AnemiaEnv(X_train, y_train, random=True)


# In[13]:


from stable_baselines3 import DQN


# In[14]:


model_dqn = DQN('MlpPolicy', environment, learning_rate=0.01,
                buffer_size=100000, learning_starts=1000, batch_size=64, gamma=0.99,
                train_freq=4, gradient_steps=1,exploration_fraction=0.7, 
                exploration_initial_eps=1.0, exploration_final_eps=0.05)


# In[15]:


model_dqn.learn(total_timesteps=10000, log_interval=4)


# In[16]:


def evaluate_dqn(dqn_model, X_test, y_test):
    test_df = pd.DataFrame(columns=['index', 'episode_length', 'reward', 'y_pred', 'y_actual', 'trajectory', 'terminated', 'is_success'])
    env = AnemiaEnv(X_test, y_test, random=False)
    count=0

    try:
        while True:
            count+=1
            obs, done = env.reset(), False
            while not done:
                action, _states = dqn_model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done == True:
                    test_df = test_df.append(info, ignore_index=True)
    except StopIteration:
        print(f'Finished evaluating the model on {count} samples')
    return test_df


# In[ ]:


test_data = pd.read_csv('data/test_set_constant.csv')   
test_data.fillna(-1, inplace=True)
X_test = test_data.iloc[:, 0:-1]
y_test = test_data.iloc[:, -1]
X_test, y_test = np.array(X_test), np.array(y_test)
test_df = evaluate_dqn(model_dqn, X_test, y_test)


# In[15]:


print(f"accuracy: {test_df['is_success'].mean()}")


# In[ ]:


import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open('model.ipynb', 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Convert to Python script
python_exporter = PythonExporter()
python_code, _ = python_exporter.from_notebook_node(notebook)

# Save as .py file
with open('model.py', 'w', encoding='utf-8') as f:
    f.write(python_code)

