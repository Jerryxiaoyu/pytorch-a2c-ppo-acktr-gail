import sys
path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if path in sys.path:
    sys.path.remove(path)

import gym
import numpy as np

from my_envs.mujoco import *
from my_envs.base.command_generator import command_generator
from matplotlib.pylab import plt
import time
import os
from utils.Logger import LoggerCsv
from eval.plot_results import *
import pandas as pd
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs

root_path = '/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr'
os.chdir(root_path)

result_dir = 'eval/results_any'
seed =122

def GA_runner(num_times):
    rand_init = 1
    action_dim = 2
    CPG_enable = 1
    reward_choice= 26
    state_mode = 'pos'
    os.environ["REWARD_CHOICE"] = str(reward_choice)
    os.environ["ACTION_DIM"] = str(action_dim)
    os.environ["CPG_ENABLE"] = str(CPG_enable)
    os.environ["STATE_MODE"] = str(state_mode)
    os.environ["RAND_INIT"] = str(rand_init)
    env = gym.make('CellrobotEnvCPG4-v0')



    command =  command_generator(10000, 0.01, 100, vx_range=(0.5, 1), vy_range = (0,0), wyaw_range = (0,0))

    render = True

    num_enjoy = 1

    #for i in range(num_times):


    env._max_episode_steps = 2000
    for i in range(num_times):
        num_episodes = 0
        logger = LoggerCsv(result_dir, csvname='log_data{}'.format(i))
        #logger = None

        env.seed(i*2+seed)
        obs = env.reset()

        velocity_base_lists = []
        command_lists = []
        reward_lists = []
        num_episodes_list = []
        obs_lists = []


        action = np.array([1,1])

        while True:
            # Obser reward and next obs


            #action = position_PID(obs , action)


            obs, reward, done, log_info = env.step(action)


            if logger is not None:
                velocity_base_lists.append(log_info['velocity_base'])
                command_lists.append(log_info['commands'])
                reward_lists.append(log_info['rewards'])
                num_episodes_list.append(num_episodes)
                obs_lists.append(log_info['obs'])

            if render:
                env.render()

            if done:
                num_episodes += 1
            if num_episodes == num_enjoy:
                break

        if logger is not None:
            velocity_base = np.array(velocity_base_lists, dtype=np.float64)
            commands = np.array(command_lists, dtype=np.float64)
            rewards = np.array(reward_lists, dtype=np.float64)
            num_episodes_lists = np.array(num_episodes_list, dtype=np.float64).reshape((-1, 1))
            obs_lists = np.array(obs_lists, dtype=np.float64)

            data = np.concatenate((num_episodes_lists, velocity_base, commands, obs_lists, rewards), axis=1)

            trajectory = {}
            for j in range(data.shape[0]):
                for i in range(data.shape[1]):
                    trajectory[i] = data[j][i]
                logger.log(trajectory)
                logger.write(display=False)


# def position_PID(obs, action ):
#     #target_angles = target_angles.reshape((-1, 1))
#
#     #q = cur_angles.reshape((-1, 1))
#
#     kp =  5
#     error = obs[1] - 0
#     # action[0] = action[0] - kp *(error)
#     # action[1] = action[1] + kp * (error)
#     #
#
#     action[0] = 1 -  kp *(error)
#     action[0] = 1 + kp * (error)
#
#     action = np.clip(action, 0, 1.5)
#     #action = np.array(action)
#
#     return action.reshape((1, -1))[0]
# # get original data from pure GA
GA_runner(1)


