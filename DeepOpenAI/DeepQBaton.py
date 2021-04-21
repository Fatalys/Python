# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 12:57:25 2021

@author: Administrator
"""

import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

model = Sequential()
model.add(Dense(50, activation='relu', input_shape=[4]))
model.add(Dense(200, activation='relu'))
model.add(Dense(2))
model.summary()    

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4))#, metrics=['accuracy'])

env = gym.make('CartPole-v0')

epoch = 10000  
#Initialise Epsilon pour un epsilon greedy
Epsilon = np.linspace(0.99,0.05, epoch)
alpha = 0.05
gamma = 0.98
Moyenne = 0

for i_episode in range(epoch):
    
    #Pour enregistrer les informations
    memory_observation = []
    memory_action = []
    memory_reward = []
    
    observation = env.reset()
    memory_observation.append(observation)
    
    if i_episode%500 == 0:
        print("Etape :", i_episode, "/", epoch)
        print("Moyenne de temps d'équilibre :", Moyenne/500)
        Moyenne = 0
    
    for t in range(200):

        env.render() if i_episode%1000 == 0 else 0

        # SELECTION DE L'ACTION
        if random.random()<Epsilon[i_episode]:
            action = random.randint(0, 1)
        else:
            action = np.argmax(model.predict(np.expand_dims(observation, axis=0)))
            
        memory_action.append(action)
        observation, reward, done, _ = env.step(action)
        memory_observation.append(observation)
        memory_reward.append([reward])
        
        if done:
            memory_reward.append([-10])
            memory_reward = memory_reward[1:]
            Moyenne += t+1
            break
  
    # Cherche la politique optimale grâce à l'équation de Bellman
        
    memory_observation = np.array(memory_observation) 
    memory_action = np.array(memory_action)
    anti_action = 1 - memory_action 
    anti_mask = tf.one_hot(anti_action,2)
    mask = tf.one_hot(memory_action,2)
    memory_reward = np.array(memory_reward) 
    Q_values = model.predict(memory_observation)
    next_Q_values = Q_values[1:] 
    Q_values = Q_values[:-1]
    
    best_next_actions = tf.math.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions,2)
    next_best_Q_values = tf.reduce_sum(next_Q_values*next_mask, axis=1, keepdims=True)
    target_Q_values = memory_reward + gamma * next_best_Q_values  
    
    total_target = target_Q_values*mask + Q_values*anti_mask #- Q_values
    
    # Entraine en cherchant à atteindre la politique optimale
    model.fit(memory_observation[:-1], total_target, steps_per_epoch=10, epochs=1, verbose=0)    
    
print("Sauvegarde du modele")
model.save("ModelBaton")
env.close()

#loss = ((target_Q_values-Q_values)*mask)**2
