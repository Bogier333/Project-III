
import numpy as np
import pprint

#states = {1,2,3,4,5,6}, actions = {up, down, left, right}



def next_state(state, action):
    if (state == 1 and action == 'right'):
        state = 2
    elif (state == 1 and action == 'down'):
        state = 6
    elif (state == 2 and action == 'left'):
        state = 1
    elif (state == 2 and action == 'down'):
        state = 3
    elif state == 3 and action == 'up':
        state = 2
    elif state == 3 and action =='left':
        state = 6
    elif state == 3 and action == 'down':
        state = 4
    elif state == 4 and action == 'up':
        state = 3
    elif state == 4 and action == 'left':
        state = 5
    return state
        
Q = {}
pairs_visited = {}

Q[(1,'right')] = 0
Q[(1,'down')] = 0
Q[(2,'left')] = 0
Q[(2,'down')] = 0
Q[(3,'up')] = 0
Q[(3,'left')] = 0
Q[(3,'down')] = 0
Q[(4,'up')] = 0
Q[(4,'left')] = 0

pairs_visited[(1,'right')] = 0
pairs_visited[(1,'down')] = 0
pairs_visited[(2,'left')] = 0
pairs_visited[(2,'down')] = 0
pairs_visited[(3,'up')] = 0
pairs_visited[(3,'left')] = 0
pairs_visited[(3,'down')] = 0
pairs_visited[(4,'up')] = 0
pairs_visited[(4,'left')] = 0

def possible_actions(state):
    if state == 1:
        return ['right','down']
    elif state == 2:
        return ['left','down']
    elif state == 3:
        return ['up', 'down', 'left']
    elif state == 4:
        return ['up', 'left']
    
def possible_Q_values(state, Q):
    actions = possible_actions(state)
    
    Q_ = {x:y for x,y in Q.items() if x[1] in actions and x[0] == state}
    return Q_

def nice_print(Q):
    for x in Q:
        print (x)
    for y in Q[x]:
        print (y,':',Q[x][y])

def eps_greedy(Q, eps, state):
    Q_ = possible_Q_values(state, Q)
    available_actions = possible_actions(state)
    rand = np.random.random()
    
    if rand < eps:
        action = np.random.choice(available_actions)
    else:
        best_state_action_pairs = [keys[1] for keys,values in Q_.items() if values == max(Q_.values())]
        
        action = np.random.choice(best_state_action_pairs)

    return action



def episode(p = False):
    done = False
    state = 1
    memory = []
    
    while not done:
        action = eps_greedy(Q, 0.1, state)
        memory.append((state, action))    
        state = next_state(state, action)
            
        if state == 6:
            reward = 1
            done = True
        elif state == 5:
            reward = 10
            done = True
            
            
    for i in memory:

        pairs_visited[i] += 1
        
        #alpha = 0.01
        alpha = (1/pairs_visited[i])

        Q[i] = Q[i] + alpha * (reward - Q[i])
            
    if p == True:
        print(memory, reward)
        



for i in range(100000):
    episode(False)


pprint.pprint(Q, width = 1)



 
    
    
    
    
    
    
    
    
    
    
    
