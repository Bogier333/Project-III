import gym
import numpy as np
import matplotlib.pyplot as plt
from blackjack_plotter import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v1')
eps = 1

Q = {}

agentSumSpace = [i for i in range(4,32)]
dealerCardSpace = [i for i in range(1,11)]
agentAceSpace = [False, True]
actionSpace = [0,1]

stateSpace = []
returns = {}
pairsVisited = {}

def eps_greedy(Q, eps, state):
    rand = np.random.random()
    
    if rand < eps:
        action = np.random.choice(actionSpace)
    else:
        values = np.array([Q[(state, 0)], Q[(state, 1)]])
        action = np.random.choice(np.flatnonzero(values == values.max()))
    
    return action

for total in agentSumSpace:
    for card in dealerCardSpace:
        for ace in actionSpace:
            for action in actionSpace:
                Q[((total, card, ace), action)] = 0
                returns[((total, card, ace), action)] = 0
                pairsVisited[((total, card, ace), action)] = 0
            stateSpace.append((total, card, ace))
        
numEpisodes = 10000000

for i in range(numEpisodes):
    observation = env.reset()
    done = False

    memory = []
    if i % 100000 == 0:
        print(i)
    while not done:
        state = (observation[0], observation[1], observation[2])
        action = eps_greedy(Q, eps, state)
        observation_, reward, done, info = env.step(action)
        

        memory.append(((observation[0], observation[1], observation[2]), action))
        pairsVisited[((observation[0], observation[1], observation[2]), action)] += 1
        
        
        observation = observation_
        
        
    pairsVisited[((observation[0], observation[1], observation[2]), action)] += 1
    
    for i in memory:
        Q[i] = Q[i] + (1/pairsVisited[i]) * (reward - Q[i])
        
    
    eps = eps*0.999999
    
plt.rcParams.update({'font.size': 10})
V={}

for x in Q.items():
    V[x[0][0]] = [Q[(x[0][0],0)], Q[(x[0][0],1)]]

V=  {x:y for x,y in V.items() if y!=[0,0]}

policy = dict((k,np.argmax(v)) for k, v in V.items())

plot_policy(policy)
        
    

        
    