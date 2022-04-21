import numpy as np
import math
import matplotlib.pyplot as plt

bandit_probs = [0.25, 0.3, 0.35, 0.4]

def slot_k(k):
    return np.random.binomial(1, bandit_probs[k])


num_episodes = 10000
n=100

bandit_0 = np.zeros((n, num_episodes))
bandit_1 = np.zeros((n, num_episodes))
bandit_2 = np.zeros((n, num_episodes))
bandit_3 = np.zeros((n, num_episodes))

def TS_action(bandit_alpha_beta):
    samples = [np.random.beta(bandit_alpha_beta[i][0], bandit_alpha_beta[i][1]) for i in range(4)]
    return np.argmax(samples)

def eps_greedy_action(Q, eps):
        if np.random.random()<eps:
            chosen_bandit = np.random.choice([0,1,2,3])
        else:
            chosen_bandit = np.random.choice(np.flatnonzero(Q == Q.max()))
        return chosen_bandit
    
    

for j in range(n):
    Q=np.zeros(4)
    eps = 0.3
    bandit_counter = [0,0,0,0]
    bandit_0_prop=[]
    bandit_1_prop=[]
    bandit_2_prop=[]
    bandit_3_prop=[]
    bandit_reward_counter=[0,0,0,0]
    bandit_alpha_beta = [[1,1],[1,1],[1,1],[1,1]]
    for i in range(num_episodes):
        
        chosen_bandit = eps_greedy_action(Q, eps)
        #chosen_bandit = TS_action(bandit_alpha_beta)
    
        bandit_counter[chosen_bandit] += 1
        draw =  slot_k(chosen_bandit)
        bandit_reward_counter[chosen_bandit] += draw
        bandit_alpha_beta[chosen_bandit] = [1 +  bandit_reward_counter[chosen_bandit], 1 + bandit_counter[chosen_bandit] - bandit_reward_counter[chosen_bandit] ]
        
        
        Q[chosen_bandit] = Q[chosen_bandit] + (1/bandit_counter[chosen_bandit]) * (draw - Q[chosen_bandit])
        
        
        bandit_0_prop.append(bandit_counter[0]/(i+1))
        bandit_1_prop.append(bandit_counter[1]/(i+1))    
        bandit_2_prop.append(bandit_counter[2]/(i+1))
        bandit_3_prop.append(bandit_counter[3]/(i+1))
        
        
        # eps -= 1e-4
        # if eps < 0 :
        #     eps = 0

    
    
    bandit_0[j,:] = bandit_0_prop
    bandit_1[j,:] = bandit_1_prop
    bandit_2[j,:] = bandit_2_prop
    bandit_3[j,:] = bandit_3_prop
    
# print(bandit_3)
    
    
band_0_total_prop = [np.mean(bandit_0[:,i]) for i in range(num_episodes)]
band_1_total_prop = [np.mean(bandit_1[:,i]) for i in range(num_episodes)]
band_2_total_prop = [np.mean(bandit_2[:,i]) for i in range(num_episodes)]
band_3_total_prop = [np.mean(bandit_3[:,i]) for i in range(num_episodes)]

# print(band_3_total_prop)
    
    

plt.plot(band_0_total_prop, label = "Bandit 1")

plt.plot(band_1_total_prop, label = "Bandit 2")

plt.plot(band_2_total_prop, label = "Bandit 3")

plt.plot(band_3_total_prop, label = "Bandit 4")

plt.legend(loc="upper right")
plt.ylim(top=1)
plt.ylabel("Proportion of Bandit")
plt.xlabel("Number of episodes")
plt.show()
    
    
    
