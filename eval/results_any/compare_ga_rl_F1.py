import numpy as np
from matplotlib.pylab import plt




import gym
import numpy as np

from my_envs.mujoco import *
from my_envs.base.command_generator import command_generator
from matplotlib.pylab import plt
import time
import os


def env_test():
    action_dim = 1
    CPG_enable = 1
    reward_choice=20
    os.environ["REWARD_CHOICE"] = str(reward_choice)
    os.environ["ACTION_DIM"] = str(action_dim)
    os.environ["CPG_ENABLE"] = str(CPG_enable)
    env = gym.make('CellrobotEnvCPG4-v0')  # Swimmer2-v2  SpaceInvaders-v0 CellrobotEnv-v0  CellrobotEnvFull-v0
    env._max_episode_steps = 2000
    #env = gym.wrappers.Monitor(env, 'tmp/tmp.mp4', force=True)

    command =  command_generator(10000, 0.01, 100, vx_range=(0.5, 1), vy_range = (0,0), wyaw_range = (0,0))

    obs = env.reset()

    render = True
    max_step = 2000


    v_e = []
    log_infos=[]

    #action = np.ones(2) * (1)
    #action[-4:] =np.array([0.5,0.8,0.5,0.8])
    action = np.array([1.0, 1.0])

    for i in range(max_step):
        if render:
             env.render()

        next_obs, reward, done, log_infos = env.step(action)



        obs = next_obs
        #log_infos.append(arg)
        v_e.append(log_infos['velocity_base'])

        #env.render(mode='rgb_array')#mode='rgb_array'
    env.close()