# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:56:13 2021

@author: Administrator
"""

import gym
import numpy as np
import random

#Crée l'environnement
env = gym.make('Acrobot-v1')

Amplitude = env.observation_space.high - env.observation_space.low

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

Table_Q = np.zeros([20,10,10,10,30,30,3])

epoch = 20000  
#Initialise Epsilon pour un epsilon greedy
Epsilon = np.linspace(0.99,0.05, epoch)
alpha = 0.1
gamma = 0.98
Reussie = 0

for i_episode in range(epoch):
    
    if i_episode%500 == 0:
        print("Etape :", i_episode, "/", epoch)
        print("Nombre de réussite :", Reussie,"/ 500")
        Reussie = 0
    
    observation = env.reset()
    old_etat = etat_discret(observation)
    new_etat = old_etat
    
    for t in range(1000):

        env.render() if i_episode%1000 == 0 else 0
        
        # SELECTION DE L'ACTION
        if random.random()<Epsilon[i_episode]:
            action = random.randint(0, 2)
        else:
            action = np.argmax(Table_Q[new_etat])
        
        observation, reward, done, info = env.step(action) 
        new_etat = etat_discret(observation)

        # MAJ DE LA TABLE Q
        Table_Q[old_etat][action] = (1-alpha)*Table_Q[old_etat][action] + alpha*(reward+gamma*max(Table_Q[new_etat]))
        old_etat = new_etat
        
        if reward == 0:
            Reussie += 1

        if done:
            break
        
np.save("Table_Pendule", Table_Q)
env.close()
