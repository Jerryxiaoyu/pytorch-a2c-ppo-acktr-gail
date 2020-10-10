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
from my_envs.utils.goals import generate_point_in_arc_area, generate_same_interval_eight_curve, generate_eight_curve

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
REACH_THRESHHOLD = 0.10

proj_dir = "/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr"

class CellRobotEnvCPG6NewP2PTarget(CellRobotEnvCPG6GoalTraj):
    def __init__(self,
                 max_steps = 2000,
                 isRenderGoal = 0,
                 sample_mode = 0,
                 hardReset_per_reset=5,
                 num_goals = 1,
                 **kwargs):

        #
        #
        # self.sample_mode = os.getenv('SAMPLE_MODE')
        # if self.sample_mode is None:
        #     self.sample_mode = sample_mode
        #     print('sample_mode is not sepecified, so sample_mode is default  {}'.format(sample_mode))
        # else:
        #     self.sample_mode = int(self.sample_mode)
        #     print('sample_mode = ', self.sample_mode)

        self.reset_count = 0
        self.hardReset_per_reset = hardReset_per_reset

        self.num_goals = num_goals

        self._isRenderGoal = isRenderGoal
        self.max_steps = max_steps

        self.goal_state = np.zeros(self.num_goals*3)
        super().__init__(**kwargs)



    #_render_goal_position
    def _goal2render(self, goal_state):
        assert  np.array(goal_state).shape[0] == 3
        return  goal_state


    def reset(self, *args):
        if self.reset_count == 0 or self.reset_count >= self.hardReset_per_reset:
            self.sim.reset()
            ob = self.reset_model(*args)
            self.reset_count = 0
        else:
            ob = self.reset_model(*args)

        self.reset_count += 1
        return ob



    def reset_model(self, command=None, ):
        if self.reset_count == 0 or self.reset_count >= self.hardReset_per_reset:
            # reset init robot
            self._reset_robot_position()

        # # sample_goal
        # self.goal_state = self.sampel_goal()


        # if self._isRenderGoal:
        #     self._render_goal_position(self._goal2render(self.goal_state))

        self.goal_orien_yaw = 0
        self.goal_x = 0
        self.goal_y = 0
        self.goal_orien_yaw = 0
        self.goal_xyyaw = np.array([0.0, 0.0, 0.0])

        for i in range(6):
            self.filter_fun_list[i].reset()

        self._sample_command(command)

        # reset something...

        if self.reset_count == 0 or self.reset_count >= self.hardReset_per_reset:
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

        if self.reset_count == 0 or self.reset_count >= self.hardReset_per_reset:
            if self.trajectory_length > 0:
                self.history_buffer.full_state(self.obs_robot_state)
        self._t_step = 0

        # reset something...
        if self.reset_count == 0 or self.reset_count >= self.hardReset_per_reset:
            self.CPG_controller = CPG_controller_fun(CPG_node_num=self.num_joint, position_vector=position_vector,
                                                     dt=self.dt,
                                                     mode=self.cpg_mode)
        self._last_root_position = self.root_position
        self._last_root_euler = self.root_euler

       # self._reset_cnt += 1
        return obs

    def _sample_command(self, command):
        if command is None:

            points = generate_eight_curve(A= 3, b=2, vel=0.1, dt = 0.05)
            self.command = np.concatenate((points, np.zeros(points.shape[0])[:, None]), axis=1)

            #print("command shape:", self.command.shape)


        else:
            self.command = command

        global_command = os.getenv('GLOBAL_CMD')
        if global_command is not None:
            self.command = IO('{}/data/cmd_{}.pkl'.format(proj_dir, global_command)).read_pickle()
            print('Global command is selected, cmd_{}'.format(global_command))

        # _render_goal_position
    def _render_goal_position(self, position):
        """
        :param position: [x,y,z]
        :return:
        """
        assert self.num_goals <= 5
        for i in range(self.num_goals):

            self.model.site_pos[i] = position.reshape((-1, 3))[i]
            if i == 0:
                self.model.site_rgba[i] = [1, 0.5, 0, 1]
            else:
                self.model.site_rgba[i] = [1, 0, 0, 1]
    def step(self, a):

        if self._isRenderGoal:
            self._render_goal_position(self.command[self._t_step:self._t_step+self.num_goals])
        obs, reward, done, info= super().step(a)
       # print('reward: ', reward)
        done, terminal_reward = self._terminal()

        #info['goal_state'] = self.goal_state
        return obs, reward + terminal_reward, done, info

    def _terminal(self):

        q_state = self.state_vector()
        notdone = np.isfinite(q_state).all() \
                  and self.get_body_com("torso")[2] >= 0.1 and self.get_body_com("torso")[2] <= 0.6
        done = not notdone

        if done:
            return True, -10
        if self._t_step > self.max_steps:
            return True, 0
        dis = np.linalg.norm(self.root_position[:2]- self.current_command[:2])
        if dis > 5:
            #print("over distance : " )
            #print("current timestep:{},  extra reward :{} ".format(self._t_step,-dis*(self.max_steps - self._t_step) ))
            return True, -1000#-dis*(self.max_steps - self._t_step)
        return False, 0



    def _get_obs(self):

        self._robot_state = self._get_robot_state()

        self.obs_robot_state = self.robot_state

        self.obs_robot_state = np.concatenate([
            self.obs_robot_state,
            self.root_quat.flatten()
        ])
        #print("root pos:", self.root_position)
        # concat history state
        if self.trajectory_length > 0:
            history_traj = self.sampled_history_trajectory

            delta_position = history_traj[:,:3] - self.root_position
            root_pos_history = np.array(
                [self.root_inv_rotation.dot(delta_position[_]) for _ in range(delta_position.shape[0])])

            obs = np.concatenate([
                self.obs_robot_state[3:],
                root_pos_history.flatten(),
                history_traj[:,3:].flatten()

            ])
        else:
            obs = self.obs_robot_state

        # concat cmd
        cmd = self._get_goal_info()
        if cmd is not None:
            obs = np.concatenate([obs,
                                  cmd])
        return obs

    def _get_goal_info(self):

        if self.command_mode == 'p2p':
            #print("goal: {}, root pos: {}".format(self.goal_state[:2], self.root_position[:2] ))

            #cmd = self.root_inv_rotation.dot(self.current_command- self.root_position)[:2] ## only x y

            root2goal_vec = []
            for i in range(self.num_goals):
                root2goal_vec.append(self.root_inv_rotation.dot(self.command[self._t_step+i] - self.root_position)[:2])
            cmd = np.array(root2goal_vec).flatten()

            #print("cmd: ", cmd)
        else:
            raise NotImplementedError
        return cmd

    def compute_reward(self, velocity_base, v_commdand, action, obs):
        if self.reward_choice == 0:
            ## line
            dis = np.linalg.norm(self.current_command[:2] - self.root_position[:2])
            forward_reward = -dis*5

            # if dis  < REACH_THRESHHOLD:
            #     forward_reward += 5

            ctrl_cost = 0#.5 * np.square(action).sum()
            contact_cost = 0#0.5 * 1e-4 * np.sum( np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = -1.0
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward

            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])

           # print('reward: ', reward)
            #print(other_rewards)

        elif self.reward_choice == 1:
            ## line
            vel = np.linalg.norm(self.last_root_position - self.current_command) - np.linalg.norm(
                self.root_position - self.current_command)
            forward_reward = vel*1000

            # if dis  < REACH_THRESHHOLD:
            #     forward_reward += 50

            ctrl_cost = 0#.5 * np.square(action).sum()
            contact_cost =0# 0.5 * 1e-4 * np.sum( np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = -1.0
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward

            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])



        else:
            raise NotImplementedError
        #print(other_rewards)
        return reward, other_rewards


#
# import transformations
#
# class CellRobotEnvCPG6NewMultiTarget(CellRobotEnvCPG6NewTarget):
#     def __init__(self,
#                  **kwargs):
#
#         self.phase_time = 0
#         self.goal_data = self.generate_curve()
#         super().__init__(**kwargs)
#
#     def reset_model(self, command=None,):
#         obs = super().reset_model(command)
#         self.phase_time = 0
#         return obs
#
#     def generate_curve(self):
#         points = generate_same_interval_eight_curve(dis=0.3)
#         goal_positions = np.concatenate((points, np.zeros(points.shape[0])[:, None]), axis=1)
#
#         return goal_positions
#
#     def _get_goal_info(self):
#         if self.command_mode == 'point':
#             #print("goal: {}, root pos: {}".format(self.goal_state[:2], self.root_position[:2] ))
#
#             #cmd = self.root_inv_rotation.dot(self.goal_state- self.root_position)[:2] ## only x y
#
#             root2goal_vec = []
#             for i in range(self.num_goals):
#                 root2goal_vec.append(self.root_inv_rotation.dot(self.goal_state.reshape((-1, 3))[i] - self.root_position)[:2])
#             cmd = np.array(root2goal_vec).flatten()
#
#             #print("cmd: ", cmd)
#         else:
#             raise NotImplementedError
#         return cmd
#
#     def sampel_goal(self):
#         if self.sample_mode == 0:
#
#             theta_list = [np.deg2rad(30), np.deg2rad(45), np.deg2rad(45), np.deg2rad(45)]
#             dis_list = [(0.1, 1), (0.1, 1), (0.1, 1), (0.1, 1)]
#             goal_points = []
#             center_p = [self.root_position[0], 0, self.root_position[1]]
#             norm_dir = [self.root_direction[0], 0, self.root_direction[1]]
#
#             for i in range(self.num_goals):
#
#                 final_p = generate_point_in_arc_area(center_p, norm_dir, theta=theta_list[i], dis_range=dis_list[i])
#
#                 goal_points.append([final_p[0], final_p[2], 0])
#
#
#         elif self.sample_mode ==1:
#
#             rand_init = 0.2
#
#             idx = np.random.randint(0, self.goal_data.shape[0])
#             goal_points = []
#             for i in range(self.num_goals+1):
#                 x = self.goal_data[(idx + i) % self.goal_data.shape[0]][0]
#                 y = self.goal_data[(idx + i) % self.goal_data.shape[0]][1]
#                 goal_points.append([x, y, 0])
#             goal_positions = np.array(goal_points)
#
#
#             goal_dir = goal_positions[1] - goal_positions[0]
#             rot = transformations.rotation_matrix(
#                 np.arctan2(self.root_direction[1], self.root_direction[0]) - np.arctan2(goal_dir[1], goal_dir[0]),
#                 [0, 0, 1])[0:3, 0:3]
#
#             goal_positions_transfered = rot.dot(goal_positions.transpose()).transpose()
#             goal_positions_transfered += (self.root_position - goal_positions_transfered[0])+ np.array([np.random.uniform(-rand_init,rand_init ),np.random.uniform(-rand_init,rand_init ), 0])
#
#             goal_points = goal_positions_transfered[1:]
#
#             ## check outspace
#             return np.array(goal_points).flatten()
#
#         else:
#             raise NotImplementedError
#         ## check outspace
#         return np.array(goal_points).flatten()
#
#     #_render_goal_position
#     def _goal2render(self, goal_state):
#
#         return  goal_state
#
#     def _render_goal_position(self, position):
#         """
#         :param position: [x,y,z]
#         :return:
#         """
#         assert  self.num_goals <= 5
#         for i in range(self.num_goals):
#
#             self.model.site_pos[i] = position.reshape((-1,3))[i]
#             if i ==0:
#                 self.model.site_rgba[i] = [1 ,0.5, 0 ,1]
#             else:
#                 self.model.site_rgba[i] = [1, 0, 0, 1]
#
#     def _terminal(self):
#
#         q_state = self.state_vector()
#         notdone = np.isfinite(q_state).all() \
#                   and self.get_body_com("torso")[2] >= 0.1 and self.get_body_com("torso")[2] <= 0.6
#         done = not notdone
#
#         if done:
#             return True
#         if self._t_step > self.max_steps:
#             return True
#
#         if self.phase_time == self.num_goals:
#             self.phase_time = 0
#             return True
#
#         # dis = np.linalg.norm(self.root_position- self.goal_state)
#         # if dis < REACH_THRESHHOLD:
#         #     return True
#         return False
#
#     def compute_reward(self, velocity_base, v_commdand, action, obs):
#         if self.reward_choice == 0:
#             return NotImplementedError
#
#         elif self.reward_choice == 3:
#             ## line
#             vel = np.linalg.norm(self.last_root_position - self.goal_state.reshape((-1, 3)), axis=1) - np.linalg.norm(
#                 self.root_position - self.goal_state.reshape((-1, 3)), axis=1)
#             forward_reward = vel*1000
#
#             dis = np.linalg.norm(self.goal_state.reshape((-1, 3))[:,:2] - self.root_position[:2],axis=1)
#
#             r = 0
#             if dis[self.phase_time] < REACH_THRESHHOLD:
#                 r += 50
#
#                 if self._isRenderGoal:
#                     self.model.site_rgba[self.phase_time] = [0, 1, 0,1]
#                     #self.model.site_pos[self.phase_time] = [0,0,0]
#
#                 self.phase_time += 1
#
#             else:
#                 r += vel[self.phase_time]
#
#             ctrl_cost = 0#.5 * np.square(action).sum()
#             contact_cost =0# 0.5 * 1e-4 * np.sum( np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
#             survive_reward = -1.0
#             reward = r + survive_reward
#
#             other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])
#             #print(other_rewards)
#
#         else:
#             raise NotImplementedError
#         #print(other_rewards)
#         return reward, other_rewards

#
# class CellRobotEnvCPG6NewMulti2Target(CellRobotEnvCPG6NewTarget):
#     def __init__(self,
#                  **kwargs):
#
#
#
#
#         print(self.goal_data.shape)
#         super().__init__(**kwargs)
#
#     def reset_model(self, command=None,):
#         obs = super().reset_model(command)
#
#         return obs
#
#     def generate_curve(self):
#         points = generate_same_interval_eight_curve(dis=0.3)
#         goal_positions = np.concatenate((points, np.zeros(points.shape[0])[:, None]), axis=1)
#
#         return goal_positions
#
#     def _get_goal_info(self):
#         if self.command_mode == 'point':
#             #print("goal: {}, root pos: {}".format(self.goal_state[:2], self.root_position[:2] ))
#
#             #cmd = self.root_inv_rotation.dot(self.goal_state- self.root_position)[:2] ## only x y
#
#             root2goal_vec = []
#             for i in range(self.num_goals):
#                 root2goal_vec.append(self.root_inv_rotation.dot(self.goal_state.reshape((-1, 3))[i] - self.root_position)[:2])
#             cmd = np.array(root2goal_vec).flatten()
#
#             #print("cmd: ", cmd)
#         else:
#             raise NotImplementedError
#         return cmd
#
#
#     def _sample_command(self, command):
#         if command is  None:
#
#             points = generate_same_interval_eight_curve(dis=0.3)
#             self.command = np.concatenate((points, np.zeros(points.shape[0])[:, None]), axis=1)
#
#
#         else:
#             self.command = command
#
#         global_command = os.getenv('GLOBAL_CMD')
#         if global_command is not None:
#             self.command = IO('{}/data/cmd_{}.pkl'.format(proj_dir, global_command)).read_pickle()
#             print('Global command is selected, cmd_{}'.format(global_command))
#
#
#
#     #_render_goal_position
#     def _goal2render(self, goal_state):
#
#         return  goal_state
#
#     def _render_goal_position(self, position):
#         """
#         :param position: [x,y,z]
#         :return:
#         """
#         assert  self.num_goals <= 5
#         for i in range(self.num_goals):
#
#             self.model.site_pos[i] = position.reshape((-1,3))[i]
#             if i ==0:
#                 self.model.site_rgba[i] = [1 ,0.5, 0 ,1]
#             else:
#                 self.model.site_rgba[i] = [1, 0, 0, 1]
#
#     def _terminal(self):
#
#         q_state = self.state_vector()
#         notdone = np.isfinite(q_state).all() \
#                   and self.get_body_com("torso")[2] >= 0.1 and self.get_body_com("torso")[2] <= 0.6
#         done = not notdone
#
#         if done:
#             return True
#         if self._t_step > self.max_steps:
#             return True
#
#         if self.phase_time == self.num_goals:
#             self.phase_time = 0
#             return True
#
#         # dis = np.linalg.norm(self.root_position- self.goal_state)
#         # if dis < REACH_THRESHHOLD:
#         #     return True
#         return False
#
#     def compute_reward(self, velocity_base, v_commdand, action, obs):
#         if self.reward_choice == 0:
#             return NotImplementedError
#
#         elif self.reward_choice == 3:
#             ## line
#             vel = np.linalg.norm(self.last_root_position - self.goal_state.reshape((-1, 3)), axis=1) - np.linalg.norm(
#                 self.root_position - self.goal_state.reshape((-1, 3)), axis=1)
#             forward_reward = vel*1000
#
#             dis = np.linalg.norm(self.goal_state.reshape((-1, 3))[:,:2] - self.root_position[:2],axis=1)
#
#             r = 0
#             if dis[self.phase_time] < REACH_THRESHHOLD:
#                 r += 50
#
#                 if self._isRenderGoal:
#                     self.model.site_rgba[self.phase_time] = [0, 1, 0,1]
#                     #self.model.site_pos[self.phase_time] = [0,0,0]
#
#                 self.phase_time += 1
#
#             else:
#                 r += vel[self.phase_time]
#
#             ctrl_cost = 0#.5 * np.square(action).sum()
#             contact_cost =0# 0.5 * 1e-4 * np.sum( np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
#             survive_reward = -1.0
#             reward = r + survive_reward
#
#             other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])
#             #print(other_rewards)
#
#         else:
#             raise NotImplementedError
#         #print(other_rewards)
#         return reward, other_rewards
#
#