import gym
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import pickle

pos_space = np.linspace(-1.2, 0.6, 20)
vel_space = np.linspace(-0.07, 0.07, 20)

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def get_state(observation):
    pos, vel =  observation
    pos_bin = int(np.digitize(pos, pos_space))
    vel_bin = int(np.digitize(vel, vel_space))

    return (pos_bin, vel_bin)

def max_action(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)

    return action

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    n_games = 40000
    alpha = 0.1
    gamma = 0.99
    eps = 1.0

    action_space = [0, 1, 2]

    states = []
    frames = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos, vel))

    Q = {}
    for state in states:
        for action in action_space:
            Q[state, action] = 0


    score = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        a = 0
        done = False
        obs = env.reset()
        state = get_state(obs)
        if i % 1000 == 0 and i > 0:
            print('episode ', i, 'score ', score, 'epsilon %.3f' % eps)
        score = 0
        while not done:
            if a == 0:
                action = np.random.choice([0,1,2]) if np.random.random() < eps \
                        else max_action(Q, state)
            if a == 1:
                action = action_
            obs_, reward, done, info = env.step(action)
            
            if i == 25000:
                frames.append(env.render(mode="rgb_array"))

            state_ = get_state(obs_)
            score += reward
            action_ = np.random.choice([0,1,2]) if np.random.random() < eps \
                    else max_action(Q, state)
            Q[state, action] = Q[state, action] + \
                    alpha*(reward + gamma*Q[state_, action_] - Q[state, action])
            state = state_
            a = 1

        eps = eps - 2/n_games if eps > 0.01 else 0.01
        
    save_frames_as_gif(frames)

    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])

    plt.xlabel("Episode")
    plt.ylabel("Average reward of past 50 episodes")
    plt.plot(mean_rewards)


    
