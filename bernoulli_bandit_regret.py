import numpy as np
import math
import matplotlib.pyplot as plt

bandit_probs = [0.25, 0.3, 0.35, 0.4]

def slot_k(k):
    return np.random.binomial(1, bandit_probs[k])


def TS_action(bandit_alpha_beta):
    samples = [np.random.beta(bandit_alpha_beta[i][0], bandit_alpha_beta[i][1]) for i in range(4)]
    return np.argmax(samples)

def eps_greedy_action(Q, eps):
        if np.random.random()<eps:
            chosen_bandit = np.random.choice([0,1,2,3])
        else:
            chosen_bandit = np.random.choice(np.flatnonzero(Q == Q.max()))
        return chosen_bandit
    



num_episodes = 10000
n=1000
all_regret = np.zeros((n, num_episodes))

for j in range(n):
    Q=np.zeros(4)
    eps = 0.1
    bandit_counter = [0,0,0,0]
    bandit_reward_counter=[0,0,0,0]
    bandit_alpha_beta = [[1,1],[1,1],[1,1],[1,1]]
    
    regret = []
    for i in range(num_episodes):
        
        chosen_bandit = eps_greedy_action(Q, eps)
        #chosen_bandit = TS_action(bandit_alpha_beta)
    
        bandit_counter[chosen_bandit] += 1
        draw =  slot_k(chosen_bandit)
        bandit_reward_counter[chosen_bandit] += draw
        bandit_alpha_beta[chosen_bandit] = [1 +  bandit_reward_counter[chosen_bandit], 1 + bandit_counter[chosen_bandit] - bandit_reward_counter[chosen_bandit] ]
        
        
        Q[chosen_bandit] = Q[chosen_bandit] + (1/bandit_counter[chosen_bandit]) * (draw - Q[chosen_bandit])
        
        
        regret.append(0.4 - bandit_probs[chosen_bandit])
    
        
    all_regret[j,:] = regret
    
    
regret_average = [np.mean(all_regret[:,i]) for i in range(num_episodes)]
plt.plot(regret_average, "blue",label = 'e_greedy')


num_episodes = 10000
n=1000
all_regret = np.zeros((n, num_episodes))

for j in range(n):
    Q=np.zeros(4)
    bandit_counter = [0,0,0,0]
    bandit_reward_counter=[0,0,0,0]
    bandit_alpha_beta = [[1,1],[1,1],[1,1],[1,1]]
    
    regret = []
    for i in range(num_episodes):
        
        #chosen_bandit = eps_greedy_action(Q, eps)
        chosen_bandit = TS_action(bandit_alpha_beta)
    
        bandit_counter[chosen_bandit] += 1
        draw =  slot_k(chosen_bandit)
        bandit_reward_counter[chosen_bandit] += draw
        bandit_alpha_beta[chosen_bandit] = [1 +  bandit_reward_counter[chosen_bandit], 1 + bandit_counter[chosen_bandit] - bandit_reward_counter[chosen_bandit] ]
        
        
        Q[chosen_bandit] = Q[chosen_bandit] + (1/bandit_counter[chosen_bandit]) * (draw - Q[chosen_bandit])
        
        
        regret.append(0.4 - bandit_probs[chosen_bandit])
    
        
    all_regret[j,:] = regret
    
    
regret_average = [np.mean(all_regret[:,i]) for i in range(num_episodes)]
plt.plot(regret_average, "red", label = "Thompson sampling")



plt.legend(loc="upper right")
plt.ylabel("Average Regret")
plt.xlabel("Number of episodes")
plt.show()
    
    
    
