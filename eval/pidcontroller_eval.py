import os
os.chdir('/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr')

import gym
import numpy as np

from my_envs.mujoco import *
from my_envs.base.command_generator import command_generator
from matplotlib.pylab import plt
import time

from utils.Logger import LoggerCsv,IO
import pandas as pd
from eval.plot_results import *
result_dir = ''
import ast
import argparse

parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--save-path', type= str, default=None, )

parser.add_argument('--data_name', type= str, default=None, )
parser.add_argument('--global_command', type= str, default=None, )
parser.add_argument('--rand_init', type= int, default=None )
parser.add_argument('--seed', type= int, default=None )
args = parser.parse_args()
save_plot_path = None
action_dim = 2
CPG_enable = 1
reward_choice = 26
vel_filtered = 1
rand_init = 0
seed = 17
global_command = 's1'


if args.data_name is None:
    data_name = '0'
else:
    data_name = args.data_name
if args.save_path is None:
    save_path = 'test'
else:
    save_path = args.save_path

if args.global_command is None:
    global_command = global_command
else:
    global_command = args.global_command
if args.rand_init is None:
    rand_init = rand_init
else:
    rand_init = args.rand_init
if args.seed is None:
    seed = seed
else:
    seed = args.seed

os.environ["GLOBAL_CMD"] = str(global_command)
os.environ["REWARD_CHOICE"] = str(reward_choice)
os.environ["ACTION_DIM"] = str(action_dim)
os.environ["CPG_ENABLE"] = str(CPG_enable)
os.environ["VEL_FILTER"] = str(vel_filtered)
os.environ["RAND_INIT"] = str(rand_init)

env = gym.make('CellrobotEnvCPG4-v0')
env.seed(seed)

command_path = 'data/cmd_{}.pkl'.format(global_command)
command = IO(command_path).read_pickle()



def str_mj_arr(arr):
    return ' '.join(['%0.3f' % arr[i] for i in range(arr.size)])

def print_contact_info(env, t ):
    d = env.unwrapped.data

    #print(d.ncon)
    geom_list =[]
    for coni in range(d.ncon):
        # print('  Contact %d:' % (coni,))
        con = d.contact[coni]
        if con.geom1 == 0:
            geom_list.append(con.geom2)
    return geom_list

def PID_controller(cur_vel, goal_vel,y_pose):
    #print('cur: {}, goal:{}'.format(cur_vel, goal_vel))
    k = 3

    a = k* (goal_vel-cur_vel)

    a = min(max(0, a), 1.0)

    k2 = 1.5
    delta = k2 * (  y_pose)
    action = np.array([a - delta, a + delta])
    action = np.clip(action, -1,1)
    return action



obs = env.reset()

render = True

logger = LoggerCsv(save_path, csvname='log_data_{}'.format(data_name))

num_enjoy = 1


velocity_base_lists = []
command_lists = []
reward_lists = []
num_episodes_lists = []
obs_lists = []
num_episodes = 0
contact_infos =[]
action_list =[]


action = np.ones(2) * (1)
#action[12] = 0.2
goal_vel = 0.10
vel = 0
max_step = 2000
for t in range(max_step):
    goal_vel = command[t,0]
    action = PID_controller(vel, goal_vel, obs[1])

    # Obser reward and next obs
    obs, reward, done, log_info = env.step(action)

    contact_info = print_contact_info(env, t)
    contact_infos.append(contact_info)

    vel = log_info['velocity_base'][0]

    if done:
        num_episodes += 1
    if logger is not None:
        velocity_base_lists.append(log_info['velocity_base'])
        command_lists.append(log_info['commands'])
        reward_lists.append(log_info['rewards'])
        num_episodes_lists.append(num_episodes)
        obs_lists.append(log_info['obs'])
        action_list.append(action[0])
    if num_episodes == num_enjoy:
        break
    if render:
        env.render()

if logger is not None:
    velocity_base = np.array(velocity_base_lists, dtype=np.float64)
    commands = np.array(command_lists, dtype=np.float64)
    rewards = np.array(reward_lists, dtype=np.float64)
    num_episodes_lists = np.array(num_episodes_lists, dtype=np.float64).reshape((-1, 1))
    obs_lists = np.array(obs_lists, dtype=np.float64)
    action_list = np.array(action_list, dtype=np.float64).reshape((-1, 1))

    data = np.concatenate((num_episodes_lists, velocity_base, commands, obs_lists, rewards, action_list), axis=1)

    trajectory = {}
    for j in range(data.shape[0]):
        for i in range(data.shape[1]):
            trajectory[i] = data[j][i]
        logger.log(trajectory)
        logger.write(display=False)


    #IO('contact_info.pkl').to_pickle(contact_infos)