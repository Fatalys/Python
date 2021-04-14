# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:45:30 2021

@author: Administrator
"""

import gym
import numpy as np
import random

#Crée l'environnement
env = gym.make('CartPole-v0')

#Intervalle des états possibles
Amplitude = env.observation_space.high - env.observation_space.low

Amplitude[1], Amplitude[3] = 10, 10

#Discrétise les états
def etat_discret(etat):
    faible = env.observation_space.low
    faible[1], faible[3] = 5, 5
    norm_etat = (etat - faible) / Amplitude
    return tuple((norm_etat * 40).astype(int))

Table_Q = np.zeros([40,40,40,40,2])

epoch = 300000  
#Initialise Epsilon pour un epsilon greedy
Epsilon = np.linspace(0.99,0.05, epoch)
alpha = 0.05
gamma = 0.98
Moyenne = 0

for i_episode in range(epoch):
    
    if i_episode%500 == 0:
        print("Etape :", i_episode, "/", epoch)
        print("Moyenne de temps d'équilibre :", Moyenne/500)
        Moyenne = 0
    
    observation = env.reset()
    old_etat = etat_discret(observation)
    new_etat = old_etat
    
    for t in range(1000):

        env.render() if i_episode%1000 == 0 else 0
        
        # SELECTION DE L'ACTION
        if random.random()<Epsilon[i_episode]:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Table_Q[new_etat])

        observation, reward, done, info = env.step(action) 
        new_etat = etat_discret(observation)

        # MAJ DE LA TABLE Q
        Table_Q[old_etat][action] = (1-alpha)*Table_Q[old_etat][action] + alpha*(reward+gamma*max(Table_Q[new_etat]))
        old_etat = new_etat

        if done:
            Moyenne += t+1
            break
        
np.save("Table_Baton", Table_Q)
env.close()
