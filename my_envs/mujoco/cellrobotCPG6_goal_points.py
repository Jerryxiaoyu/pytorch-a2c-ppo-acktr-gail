import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env
from math import pi, sin, cos
from transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion
from CPG_controllers.controllers.CPG_controller_quadruped import CPG_network_Sinusoid
from CPG_controllers.CPG_process  import  position_PID
from my_envs.base.ExperienceDataset import TrajectoryBuffer
from my_envs.base.command_generator import command_generator, plot_command
import os
from utils.fir_filter import fir_filter
import math
import time
from .cellrobotCPG6_goal_SMC import CellRobotEnvCPG6GoalTraj

state_M = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])

position_vector = np.array([0.9922418358258432, -0.26790716313078566, 1.544827488736095, 0.1697636863918297, 1.7741507083847519, 0.128810963574171, 0.9526971501553204, 1.8825515998993296, 1.7745743229887139, 1.0339288488706027, 1.0159186128367077, 0.6280555987489267, 1.6581479953809704, -1.7832213538976736, 0.01704474730114954, 0.0, 0.0022918597671406863, -0.02634337338777438, 0.004503538681869408, 0.0032371806499659804, 0.0, -0.03838697831725827, 0.0, 0.0, 0.0, 0.04910381476502241, 0.0, -1.0773638647322994, -1.8049011801072816, 2.4889661572487243, 1.0395144002763324, 1.8340430909060688, -2.3262172061379927, 0.7419174909731787,
                            -0.7273188675247564, -2.397205732544516, -1.460001220824175, 2.212927483411068, -2.5633159512167834, -0.9789777531415957])
joint_index = obs_low = 6
obs_high = 19

CPG_controller_fun  = CPG_network_Sinusoid
from gym.utils import seeding
from my_envs.base.global_config import *
from utils.Logger import IO
REACH_THRESHHOLD = 0.1

proj_dir = "/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr"

class CellRobotEnvCPG6Target(CellRobotEnvCPG6GoalTraj):
    def __init__(self,
                 max_steps = 2000,
                 isRenderGoal = 0,
                 sample_mode = 0,
                 **kwargs):

        self.num_goals = 1
        self.sample_mode = sample_mode
        self._isRenderGoal = isRenderGoal
        self.max_steps = max_steps


        self.goal_state = np.zeros(self.num_goals*3)
        super().__init__(**kwargs)

    def sampel_goal(self):
        if self.sample_mode == 0:
            goal_points = []
            for i in range(self.num_goals):
                x = self.root_position[0] + np.random.uniform(-3, 3)
                y = self.root_position[1] + np.random.uniform(-3, 3)
                goal_points.append([x, y, 0])
        else:
            raise NotImplementedError
        ## check outspace
        return np.array(goal_points).flatten()

    #_render_goal_position
    def _goal2render(self, goal_state):
        assert  np.array(goal_state).shape[0] == 3
        return  goal_state

    def reset_model(self, command=None, ):


        # reset init robot
        self._reset_robot_position()

        # sample_goal
        self.goal_state = self.sampel_goal()

        if self._isRenderGoal:
            self._render_goal_position(self._goal2render(self.goal_state))

        self.goal_orien_yaw = 0
        self.goal_x = 0
        self.goal_y = 0
        self.goal_orien_yaw = 0
        self.goal_xyyaw = np.array([0.0, 0.0, 0.0])

        for i in range(6):
            self.filter_fun_list[i].reset()

        self._sample_command(command)

        # reset something...
        if self.trajectory_length > 0:
            self.history_buffer = TrajectoryBuffer(num_size_per=self.robot_state_dim,
                                                   max_trajectory=self.trajectory_length)

        # set curriculum params...
        self.k0 = set_curriculum(self.curriculum_init)
        self.kd = self.kd_init
        if get_iter_rl() == self.cur_itr_rl:
            self.kc = self.kc
        elif get_iter_rl() == self.cur_itr_rl + 1:
            self.kc = math.pow(self.kc, self.kd)
            self.cur_itr_rl += 1
        else:
            raise Exception('currrent itr seems incorrect')

        obs = self._get_obs()

        if self.trajectory_length > 0:
            self.history_buffer.full_state(self.obs_robot_state)
        self._t_step = 0

        # reset something...
        self.CPG_controller = CPG_controller_fun(CPG_node_num=self.num_joint, position_vector=position_vector,
                                                 dt=self.dt,
                                                 mode=self.cpg_mode)
        self._last_root_position = self.root_position
        self._last_root_euler = self.root_euler

        self._reset_cnt += 1
        return obs


    def step(self, a):
        obs, reward, done, info= super().step(a)

        done = self._terminal()
        return obs, reward, done, info

    def _terminal(self):

        q_state = self.state_vector()
        notdone = np.isfinite(q_state).all() \
                  and self.get_body_com("torso")[2] >= 0.1 and self.get_body_com("torso")[2] <= 0.6
        done = not notdone

        if done:
            return True
        if self._t_step > self.max_steps:
            return True
        dis = np.linalg.norm(self.root_position- self.goal_state)
        if dis < REACH_THRESHHOLD:
            return True
        return False

    def _get_robot_state(self):
        joint_position = state_M.dot(self.sim.data.qpos[7:].reshape((-1, 1))).flatten()
        joint_velocity = state_M.dot(self.sim.data.qvel[6:].reshape((-1, 1))).flatten()

        root_position = self.get_body_com("torso").flatten()
        root_euler = self.get_orien()

        #print('quat', self.root_quat)
        #print(root_euler)
        root_velocity = (root_position - self.last_root_position) / self.dt

        root_angluar_velocity = (root_euler - self.last_root_euler) / self.dt

        if self.vel_filtered:
            vx_f = np.array([self.filter_fun_list[0].apply(root_velocity[0])])
            vy_f = np.array([self.filter_fun_list[1].apply(root_velocity[1])])
            vz_f = np.array([self.filter_fun_list[2].apply(root_velocity[2])])
            filterd_root_velocity = np.concatenate((vx_f, vy_f, vz_f))

            wx_f = np.array([self.filter_fun_list[3].apply(root_angluar_velocity[0])])
            wy_f = np.array([self.filter_fun_list[4].apply(root_angluar_velocity[1])])
            wz_f = np.array([self.filter_fun_list[5].apply(root_angluar_velocity[2])])
            filterd_root_angluar_velocity = np.concatenate((wx_f, wy_f, wz_f))

            robot_state = np.concatenate([
                root_position,  # 3
                #self.root_rotation.flatten(),#9
                root_euler,  # 3
                joint_position,  # 13
                joint_velocity,  # 13
                # root_velocity,  # 3
                # root_angluar_velocity,  # 3
                filterd_root_velocity,
                filterd_root_angluar_velocity,
            ])
        else:
            robot_state = np.concatenate([
                root_position,  # 3

                root_euler,  # 3
                joint_position,  # 13
                joint_velocity,  # 13
                root_velocity,  # 3
                root_angluar_velocity,  # 3

            ])
        return robot_state

    def _get_obs(self):

        self.obs_robot_state = self._robot_state[6:]

        self.obs_robot_state = np.concatenate([
            self.obs_robot_state,
            self.root_rotation.flatten()
        ])

        # concat history state
        if self.trajectory_length > 0:
            obs = np.concatenate([
                self.obs_robot_state,
                self.sampled_history_trajectory.flatten()
            ])
        else:
            obs = self.obs_robot_state

        # concat cmd
        cmd = self._get_goal_info()
        if cmd is not None:
            obs = np.concatenate([obs,
                                  cmd])

        # pred_position = self._get_pred_root_position()
        # obs = np.concatenate([obs,
        #                       pred_position])
        return obs

    def _get_goal_info(self):
        if self.command_mode == 'point':
            #print("goal: {}, root pos: {}".format(self.goal_state[:2], self.root_position[:2] ))
            cmd = np.linalg.inv(self.root_rotation).dot(self.goal_state- self.root_position)[:2] ## only x y

            #print("cmd: ", cmd)
        else:
            raise NotImplementedError
        return cmd

    def compute_reward(self, velocity_base, v_commdand, action, obs):
        if self.reward_choice == 0:
            ## line
            dis = np.linalg.norm(self.goal_state[:2] - self.root_position[:2])
            forward_reward = -dis

            if dis  < REACH_THRESHHOLD:
                forward_reward += 50

            ctrl_cost = .5 * np.square(action).sum()
            contact_cost = 0.5 * 1e-4 * np.sum(
                np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = -1.0
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward

            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])

            #print(other_rewards)
        elif self.reward_choice == 1:
            ## line
            dis = np.linalg.norm(self.goal_state[:2] - self.root_position[:2])
            forward_reward = -dis

            # if dis  < REACH_THRESHHOLD:
            #     forward_reward += 50

            ctrl_cost = .5 * np.square(action).sum()
            contact_cost = 0.5 * 1e-4 * np.sum(
                np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = -1.0
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward

            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])
        else:
            raise NotImplementedError
        #print(other_rewards)
        return reward, other_rewards

