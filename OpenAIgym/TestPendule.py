# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:26:43 2021

@author: Administrator
"""

import gym
import numpy as np

#Crée l'environnement
env = gym.make('Acrobot-v1')

Amplitude = env.observation_space.high - env.observation_space.low
#Amplitude[4], Amplitude[5] = 15, 20

#Discrétise les états
def etat_discret(etat):
    faible = env.observation_space.low
    norm_etat = (etat - faible) / Amplitude
    norm_etat[0] *= 20
    norm_etat[1] *= 10
    norm_etat[2] *= 10
    norm_etat[3] *= 10
    norm_etat[4] *= 30
    norm_etat[5] *= 30
    return tuple((norm_etat).astype(int))

Table_Q = np.load("Table_Pendule.npy")

Reussie = 0

for i_episode in range(10):
    
    observation = env.reset()
    new_etat = etat_discret(observation)
    done = False
    
    while not done:
        
        env.render()
        
        # SELECTION DE L'ACTION
        action = np.argmax(Table_Q[new_etat])

        observation, reward, done, info = env.step(action) 
        new_etat = etat_discret(observation)
        
        if reward == 0:
            Reussie += 1
            print("Réussie")
        
     
print("Nombre de réussite :", Reussie, "/ 10")

env.close()