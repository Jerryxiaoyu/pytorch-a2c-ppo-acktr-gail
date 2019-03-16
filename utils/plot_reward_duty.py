import os

#os.chdir('/home/drl/PycharmProjects/rl_baselines/baselines')
import gym
import numpy as np

from my_envs.mujoco import *
from my_envs.base.command_generator import command_generator
from matplotlib.pylab import plt
from evaluate.plot_results import *



def plot_all_curve(reward_fun, env_name = 'CellrobotEnvCPG2-v0', save_plot_path=None, seed = None):
    env = gym.make(env_name)  # Swimmer2-v2  SpaceInvaders-v0 CellrobotEnv-v0

    print('state: ', env.observation_space)
    print('action: ', env.action_space)

    command = command_generator(10000, 0.01, 2, vx_range=(-0.2, 0.2), vy_range=(0, 0), wyaw_range=(0, 0),   render=True, seed=seed )

    obs = env.reset(command, reward_fun)

    max_step = 1000

    v_e = []
    c_command = []
    xyz = []

    rewards = []
    action = np.ones(39) * (1)
    for i in range(max_step):
        # env.render()
        # action = env.action_space.sample()

        next_obs, reward, done, infos = env.step(action)
        obs = next_obs

        v_e.append(infos['velocity_base'])
        c_command.append(infos['commands'])
        xyz.append(infos['obs'][:3])
        rewards.append(infos['rewards'])

        # env.render(mode='rgb_array')#mode='rgb_array'
    env.close()
    dt = 0.01

    v_e = np.array(v_e)
    c_command = np.array(c_command)
    xyz = np.array(xyz)
    rewards = np.array(rewards)

    plot_velocity_curve(v_e, c_command, max_step, dt=0.01, figsize=(8, 6), save_plot_path=save_plot_path)
    plot_position_time(xyz, max_step, dt=0.01, figsize=(8, 6), save_plot_path=save_plot_path)
    plot_traj_xy(xyz, max_step, dt=0.01, figsize=(8, 6), save_plot_path=save_plot_path)

    return  rewards, command

def plot_fitness_t(reward_fun, env_name = 'CellrobotEnvCPG2-v0', save_plot_path=None, seed = None):
    env = gym.make(env_name)  # Swimmer2-v2  SpaceInvaders-v0 CellrobotEnv-v0

    print('state: ', env.observation_space)
    print('action: ', env.action_space)

    command = command_generator(10000, 0.01, 2, vx_range=(-0.2, 0.2), vy_range=(0, 0), wyaw_range=(0, 0),   render=False, seed=seed )

    obs = env.reset(command, reward_fun)

    max_step = 1000

    v_e = []
    c_command = []
    xyz = []

    rewards = []
    action = np.ones(39) * (1)

    # np.random.seed(0)
    # action = np.random.uniform(-1,1,39)
    for i in range(max_step):
        #env.render()
        #action = env.action_space.sample()

        next_obs, reward, done, infos = env.step(action)
        obs = next_obs

        v_e.append(infos['velocity_base'])
        c_command.append(infos['commands'])
        xyz.append(infos['obs'][:3])
        rewards.append(infos['rewards'])

        # env.render(mode='rgb_array')#mode='rgb_array'
    env.close()
    dt = 0.01

    v_e = np.array(v_e)
    c_command = np.array(c_command)
    xyz = np.array(xyz)
    rewards = np.array(rewards)

    plot_position_time(xyz,max_step,  save_plot_path=save_plot_path)
    plot_traj_xy(xyz, max_step, save_plot_path=save_plot_path)
    plot_velocity_curve(v_e, c_command, max_step, save_plot_path=save_plot_path)

    plt.figure(figsize=(18, 6))
    rewards_duty = []
    for i in range(rewards.shape[1] - 1):
        rewards_duty.append(np.abs(rewards[:, 1 + i] / rewards[:, 0]) * np.sign(rewards[:, 1 + i]))

        plt.plot(rewards_duty[i], label=str(i))

    plt.title('Fitness No:{}\n'
              '0:lin_vel_r,1: ang_vel_r, 2:torque_r, 3:joint_speed_r,4:orientation_r, 5:smoothness_r, 6:survive_reward\n'
              '0: forward_reward, 1:ctrl_cost, 2:  contact_cost, 3: survive_reward'.format(reward_fun))
    plt.grid()
    plt.legend()


    if save_plot_path is not None:
        plt.savefig(save_plot_path + '_fitness_duty.jpg')
    else:
        plt.show()

    return  rewards, command, v_e

#plot_fitness_t(1)