
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


V = {}
visited = {}


actual_values = [20/11, 29/11, 38/11, 74/11, 0, 0]

def MSE(list1, list2):
    n = len(list1)
    MSE = 0 
    for i in range(n):
        MSE += (list1[i] - list2[i])**2
    
    return MSE/n

def possible_actions(state):
    if state == 1:
        return ['right','down']
    elif state == 2:
        return ['left','down']
    elif state == 3:
        return ['up', 'down', 'left']
    elif state == 4:
        return ['up', 'left']
    
def rand_action(state):
    poss_actions = possible_actions(state)
    action = np.random.choice(poss_actions)
    return action


def episode_MC():
    done = False
    state = 1
    memory = []

    
    while not done:
        action = rand_action(state)
        memory.append(state)
        state = next_state(state, action)
            
        if state == 6:
            reward = 1
            done = True
        elif state == 5:
            reward = 10
            done = True
            
    
    for i in memory:
        visited[i] += 1
        # alpha = 1/visited[i]
        alpha = 0.1
        
        V[i] = V[i] + alpha*(reward - V[i])
        
def episode_TD():
    done = False
    state = 1
    memory = []
    

    while not done:
        action = rand_action(state)

        
        new_state = next_state(state, action)
            
        if new_state == 6:
            reward = 1
            done = True
        elif new_state == 5:
            reward = 10
            done = True
        else:
            reward = 0
            
        memory.append([state, reward])
        
        state = new_state
    
    memory.append([state, 0])
            
            
    
    for i in range((len(memory)-1)):
        state, reward = memory[i]
        visited[state] += 1
        # alpha = 1/visited[state]
        alpha = 0.1
        
        
        TD_target = reward + V[memory[i+1][0]]
        
        V[state] = V[state] + alpha*(TD_target - V[state])
        
    
simulations = 50
num_episodes = 1000

errors_TD = np.zeros((simulations, num_episodes))
for j in range(simulations):    
    # V[1] = 1
    # V[2] = 2
    # V[3] = 3
    # V[4] = 7
    # V[5] = 0
    # V[6] = 0
    
    V[1] = 0
    V[2] = 0
    V[3] = 0
    V[4] = 0
    V[5] = 0
    V[6] = 0

    visited[1] = 0
    visited[2] = 0
    visited[3] = 0
    visited[4] = 0
    for i in range(num_episodes):
        episode_TD()
        errors_TD[j,i] = (MSE(actual_values, list(V.values())))
        
mean_errors_TD = [np.mean(errors_TD[:,i]) for i in range(num_episodes)]    
plt.plot(mean_errors_TD, 'red', label = 'TD(0)')


errors_MC = np.zeros((simulations, num_episodes))  
for j in range(simulations):    
    # V[1] = 1
    # V[2] = 2
    # V[3] = 3
    # V[4] = 7
    # V[5] = 0
    # V[6] = 0
    
    V[1] = 0
    V[2] = 0
    V[3] = 0
    V[4] = 0
    V[5] = 0
    V[6] = 0

    visited[1] = 0
    visited[2] = 0
    visited[3] = 0
    visited[4] = 0
    for i in range(num_episodes):
        episode_MC()
        errors_MC[j,i] = (MSE(actual_values, list(V.values())))
        
mean_errors_MC = [np.mean(errors_MC[:,i]) for i in range(num_episodes)]
        
plt.plot(mean_errors_MC, 'blue', label = "Monte Carlo")

plt.legend(loc="upper right")
plt.ylabel("Mean Square Error")
plt.xlabel("Episode")
plt.show()
    



 
    
    
    
    
    
    
    
    
    
    
    