import gymnasium as gym
from stable_baselines3 import SAC
import itertools
import os
import yaml

###### FILES ######

models_dir = "models/SAC"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

###### MAIN ######

mode = input("Train ? (y/n): ")

env = gym.make('BipedalWalker-v3', render_mode="human" if mode == 'n' else 'human', hardcore=True)

N = 20
n_steps = 0
n_episodes = 0

t_steps = 10000

if mode == 'y':
    #Agent = SAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    Agent = SAC.load(f"{models_dir}/SAC", env=env)
    Agent.learning_rate = 0.0003

    for i in range(n_episodes) if n_episodes > 0 else itertools.count():
        observation, _ = env.reset()
        done = False
        while not done:
            Agent.learn(t_steps, reset_num_timesteps=False, tb_log_name="SAC")
            Agent.save(f"{models_dir}/SAC")
else:
    Agent = SAC.load(f"{models_dir}/SAC", env=env)
    for i in range(n_episodes) if n_episodes > 0 else itertools.count():
        observation, _ = env.reset()
        done = False
        score_history = []
        while not done:
            action, _ = Agent.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            score_history.append(reward)
        avg_score = sum(score_history) / len(score_history)
        print(f"Episode: {i}, Reward: {avg_score}")