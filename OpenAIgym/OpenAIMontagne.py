# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:09:54 2021

@author: Administrator
"""

import gym
import numpy as np
import random

#Crée l'environnement
env = gym.make('MountainCar-v0')

#Intervalle des états possibles
Amplitude = env.observation_space.high - env.observation_space.low

#Discrétise les états
def etat_discret(etat):
    norm_etat = (etat - env.observation_space.low) / Amplitude
    return tuple((norm_etat * 50).astype(int))

#Initialise la table Q
Table_Q = np.zeros([50,50,3])


epoch = 30000  
#Initialise Epsilon pour un epsilon greedy
Epsilon = np.linspace(0.99,0.05, epoch)
alpha = 0.1
gamma = 0.98
fini = 0

for i_episode in range(epoch):
    
    if i_episode%500 == 0:
        print("Etape :", i_episode, "/", epoch)
        print("Nombre de réussite :", fini, "/500")
        fini = 0
    
    observation = env.reset()
    old_etat = etat_discret(observation)
    new_etat = old_etat
    done = False
    
    while not done:
        
        env.render() if i_episode%1000 == 0 else 0
        
        # SELECTION DE L'ACTION
        if random.random()<Epsilon[i_episode]:
            action = random.randint(0, 2)
        else:
            action = np.argmax(Table_Q[new_etat])

        observation, reward, done, info = env.step(action) 
        new_etat = etat_discret(observation)
        
        if observation[0] >= env.goal_position:
            fini += 1
            reward = 1
        
        # MAJ DE LA TABLE Q
        Table_Q[old_etat][action] = (1-alpha)*Table_Q[old_etat][action] + alpha*(reward+gamma*max(Table_Q[new_etat]))
        old_etat = new_etat

np.save("Table_Montagne", Table_Q)
env.close()











