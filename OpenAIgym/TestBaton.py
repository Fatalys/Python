# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:36:49 2021

@author: Administrator
"""

import gym
import numpy as np

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

Table_Q = np.load("Table_Baton.npy")

Moyenne = 0

for i_episode in range(10):
    
    observation = env.reset()
    new_etat = etat_discret(observation)
    done = False
    t = 0
    
    while not done:
        
        env.render()
        t += 1
        
        # SELECTION DE L'ACTION
        action = np.argmax(Table_Q[new_etat])

        observation, reward, done, info = env.step(action) 
        new_etat = etat_discret(observation)
        
        if done:
            print("Vous avez tenu :", t, "instants")
            Moyenne += t
        
     
print("En moyenne vous avez tenu :", Moyenne/10, "instants")

env.close()