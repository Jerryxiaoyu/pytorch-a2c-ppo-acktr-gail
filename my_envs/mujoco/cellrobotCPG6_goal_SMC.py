import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env
from math import pi, sin, cos
from transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion
import transformations
from CPG_controllers.controllers.CPG_controller_quadruped import CPG_network_Sinusoid
from CPG_controllers.CPG_process  import  position_PID
from my_envs.base.ExperienceDataset import TrajectoryBuffer
from my_envs.base.command_generator import command_generator, plot_command

import os
from utils.fir_filter import fir_filter
import math
import time
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

proj_dir = "/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr"

class CellRobotEnvCPG6Goal(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    This is a new env for SMC journal
    """
    def __init__(self,
                 control_skip = 1,
                 cpg_mode = 0,
                 cpg_enable = 1,
                 robot_state_dim = 38,
                 isRootposNotInObs = False,

                 isAddDisturbance= False,
                 disturb_time_list = [100, 200, 300, 400],
                 xml_name='cellrobot_Quadruped_float_limit.xml',

                 isRenderDir = False,
                 is_hard_reset_position = False,
                 max_steps = 2000,

                 isRenderTrajectory = False
                 ):

        self.max_steps = max_steps
        self.isRenderDir = isRenderDir
        self.is_hard_reset_position = is_hard_reset_position

        self.isAddDisturbance = isAddDisturbance
        self.disturb_time_list = disturb_time_list
        self.disturb_cnt = 0

        self.isRenderTrajectory = isRenderTrajectory

        print('test cpg2........................')
        self.reward_choice = os.getenv('REWARD_CHOICE')
        self.cpg_mode = cpg_mode

        self.CPG_enable = cpg_enable

        self.num_buffer = os.getenv('NUM_BUFFER')
        self.command_mode = os.getenv('COMMAND_MODE')


        self.action_dim = os.getenv('ACTION_DIM')

        self.command_vx_high = os.getenv('COMMAND_X')
        self.command_vy_high = os.getenv('COMMAND_Y')
        self.command_wz_high = os.getenv('COMMAND_Z')

        self.command_vx_low = os.getenv('COMMAND_X_LOW')
        self.command_vy_low = os.getenv('COMMAND_Y_LOW')

        self.vel_filtered = os.getenv('VEL_FILTER')
        self.turing_flag = os.getenv('TURING_FLAG')

        self.xml_name = os.getenv('XML_NAME')

        if self.num_buffer is None:
            self.num_buffer = 1
            print('num_buffer is not sepecified, so num_buffer is default  1.')
        else:
            self.num_buffer = int(self.num_buffer)
            print('num_buffer = ',self.num_buffer )

        if self.command_mode is None:
            self.command_mode = 'no'
            print('command_mode is not sepecified, so command_mode is default  delta.')
        else:
            print('command_mode = ', self.command_mode)

        if self.CPG_enable is None:
            self.CPG_enable = 0
            print('CPG_enable is not sepecified, so CPG_enable is False 0.')
        else:
            self.CPG_enable = int(self.CPG_enable)
            print('CPG_enable = ', self.CPG_enable)

        if self.CPG_enable == 0:
            self.action_dim = None

        if self.action_dim is None:
            self.custom_action_space =  None
            print('ACTION_DIM is not sepecified, so action dim is default .')
        else:
            self.custom_action_space =  int(self.action_dim)
            print('ACTION_DIM = ', self.custom_action_space)

        if self.command_vx_high is None:
            self.command_vx_high = 0.5
            print('command_vx_high is not sepecified, so command_vx_high = 0.5 .')
        else:
            print('command_vx_high = ', self.command_vx_high)

        if self.command_vy_high is None:
            self.command_vy_high = 0
            print('command_vy_high is not sepecified, so command_vy_high = 0 .')
        else:
            print('command_vy_high = ', self.command_vy_high)


        if self.command_vx_low is None:
            self.command_vx_low = 0.
            print('command_vx_low is not sepecified, so command_vx_low = 0..')
        else:
            print('command_vx_low = ', self.command_vx_low)

        if self.command_vy_low is None:
            self.command_vy_low = 0
            print('command_vy_low is not sepecified, so command_vy_low = 0 .')
        else:
            print('command_vy_low = ', self.command_vy_low)


        if self.command_wz_high is None:
            self.command_wz_high = 0
            print('command_wz_high is not sepecified, so command_wz_high = 0 .')
        else:
            print('command_wz_high = ', self.command_wz_high)



        if self.reward_choice is None:
            self.reward_choice = 0
            print('reward_choice is not sepecified, so reward_choice is False 0.')
        else:
            self.reward_choice = int(self.reward_choice)
            print('reward_choice = ', self.reward_choice)

        if self.vel_filtered is None:
            self.vel_filtered = 0
            print('vel_filtered is not sepecified, so vel_filtered is False 0.')
        else:
            self.vel_filtered = int(self.vel_filtered)
            print('vel_filtered = ', self.vel_filtered)

        if self.turing_flag is None:
            self.turing_flag = 0
            print('turing_flag is not sepecified, so vel_filtered is False 0.')
        else:
            self.turing_flag = int(self.turing_flag)
            print('turing_flag = ', self.turing_flag)

        self.rand_init = os.getenv('RAND_INIT')
        if self.rand_init is None:
            self.rand_init = 0
        else:
            self.rand_init = int(self.rand_init)

        if self.xml_name is None:
            self.xml_name = xml_name

            self.model_path = 'cellrobot/'+self.xml_name
            print('model path:', self.model_path)
        else:

            self.model_path = 'cellrobot/' + self.xml_name
            print('model path:', self.model_path)

        print('rand_init = ', self.rand_init)

        self.curriculum_init = 0.3
        self.kd_init = 0.997

        self.k0 = set_curriculum(self.curriculum_init)
        self.kd = self.kd_init
        self.kc = self.k0
        self.cur_itr_rl = 0

        self.num_joint = 13
        policy_a_dim = 13  # networt output
        self.command = command_generator(10000, 0.01, 2, render = False)
        self._t_step = 0
        self.c_index_max = 10000
        self.action_pre = 0

        self.goal_orien_yaw = 0
        self.goal_x = 0
        self.goal_y = 0
        self.goal_xyyaw = np.array([0.0, 0.0,0.0])


        self._robot_state = np.zeros(robot_state_dim)
        self._last_root_position = np.zeros(3)
        self._last_root_euler = np.zeros(3)

        self.command_vx_low = float(self.command_vx_low)
        self.command_vx_high = float(self.command_vx_high)
        self.command_vy_low = float(self.command_vy_low)
        self.command_vy_high = float(self.command_vy_high)
        self.command_wz_low = -float(self.command_wz_high)
        self.command_wz_high = float(self.command_wz_high)

        self.command_max_step = 20000  # steps
        self.command_duration = 5 # second

        if self.turing_flag == 1:
            self.command_vx_low = 0.5
            self.command_duration = 20
        elif self.turing_flag == 2:

            self.command_duration = 5
        elif self.turing_flag == 3:

            self.command_duration = 20
        elif self.turing_flag == 4:

            self.command_duration = 4
        elif self.turing_flag == 5:
            self.command_vx_low = 0.5
            self.command_vy_low = 0.05
            self.command_duration = 20

        dt = 0.01
        self.CPG_controller = CPG_controller_fun(CPG_node_num=self.num_joint,
                                                 position_vector=position_vector, dt= dt,
                                                 mode = self.cpg_mode)


        self.robot_state_dim = robot_state_dim

        self.isRootposNotInObs = isRootposNotInObs
        if self.isRootposNotInObs:
            self.robot_state_dim -=3

        self.obs_robot_state = np.zeros(self.robot_state_dim)
        self.trajectory_length = self.num_buffer
        if self.trajectory_length >0:
            self.history_buffer = TrajectoryBuffer(num_size_per=self.robot_state_dim, max_trajectory=self.trajectory_length)

        cutoff_hz = 10 #Hz
     #   self.reward_fir= fir_filter(int(1/dt),cutoff_hz,10)


        self.filter_fun_list  = [
            fir_filter(100, 0.01, 30), #x
            fir_filter(100, 0.1, 30),   #y
            fir_filter(100, 0.1, 30),  # y
            fir_filter(100, 0.1, 30),  # wx
            fir_filter(100, 0.1, 30),  # wy
            fir_filter(100, 0.1, 30),   #wz
        ]

        # old path :'cellrobot/cellrobot_Quadruped_float_simple.xml

        self._reset_cnt = 0
        self.control_skip = control_skip

        mujoco_env.MujocoEnv.__init__(self, self.model_path,  self.control_skip , custom_action_space= self.custom_action_space)  # cellrobot_test_gen  CR_quadruped_v1_A001  'cellrobot/cellrobot_test_gen.xml' Atlas_v5/atlas_v5.xml
        utils.EzPickle.__init__(self)
        seed = mujoco_env.MujocoEnv.seed(self, 1234)

        print('seed = ', seed)
        np.random.seed(seed[0]>>32)
        print('1111REWARD_CHOICE = ', os.getenv('REWARD_CHOICE'))
        #seeding.np_random(seed[0])

        print('State size :', self.observation_space.shape[0])
        print('Policy action size : ', self.action_space.shape[0] )

    def get_one_disturbance(self):
        v_bullet = np.random.uniform(5,5)

        x = np.random.uniform(self.root_position[0] - 1, self.root_position[0] + 1)
        y = np.random.uniform(self.root_position[1] - 1, self.root_position[1] + 1)
        z = np.random.uniform(1, 2)
        p_0 = np.array([x, y, z])
        #
        # x = np.random.uniform(self.root_position[0] - 0.03, self.root_position[0] + 0.03)
        # y = np.random.uniform(self.root_position[1] - 0.03, self.root_position[1] + 0.03)
        # z = np.random.uniform(self.root_position[1] - 0.1, self.root_position[1])
        # p_target = np.array([x, y, z])
        #

        t = np.sqrt(p_0[1] ** 2 / abs((v_bullet ** 2 - np.linalg.norm(self.root_position[:2]) ** 2)))
        p_target = self.root_position + np.array([self.root_velocity[0], self.root_velocity[1], 0]) * t


        v = ( p_target -p_0 ) / np.linalg.norm(p_target - p_0) * v_bullet

        return p_0, v

    def addDisturbance(self):
        if self._t_step == self.disturb_time_list[self.disturb_cnt]:
            qpos = self.sim.data.qpos.ravel().copy()
            qvel = self.sim.data.qvel.ravel().copy()

            p_target, v = self.get_one_disturbance()

            # x = np.random.uniform(self.root_position[0] - 0.05, self.root_position[0] + 0.05)
            # y = np.random.uniform(self.root_position[1] - 0.05, self.root_position[1] + 0.05)
            # z = np.random.uniform(0.5, 1)
            # p_target =  np.array([x, y, z])#np.array([x, y, z])
            # v = [0,0,0]

            qpos[20:23] = p_target
            qvel[19:22] =  v

            self.set_state(qpos, qvel)

            self.disturb_cnt += 1

        if self.disturb_cnt == len(self.disturb_time_list):
            self.disturb_cnt = 0
    def RenderTrajectory(self):
        #print("shape :", self.model.site_pos.shape)
        # if self._t_step%5 ==0:
        #     #self.sim.model.site_pos[(self._t_step//5+5)%20000] = self.root_position
        #     self.sim.model.site_pos[(self._t_step // 5 + 5) % 20000] = self.root_position
        #     #print(self._t_step%5, self.model.site_pos[(self._t_step+5)%20000])

        root_pos = self.root_position.copy()
        root_pos[2] = 0.05


        self.sim.model.site_pos[(self._t_step  + 5) % 20000] = root_pos#self.root_position

        #
        #
        # self.sim.model.site_pos[self._t_step  + 5 + 10000] = ref_pos
        # self.sim.model.site_rgba[self._t_step + 5 + 10000] = [1, 0, 1, 1]


    def step(self, a):

        action = a.copy()

        v_commdand = self.command[self._t_step, :3]
        # self.goal_orien_yaw += v_commdand[-1]*self.dt
        self.goal_x += v_commdand[0]*self.dt
        self.goal_y += v_commdand[1]*self.dt
        # self.goal_xyyaw += v_commdand*self.dt

        if self.CPG_enable == 1 :
            action = self.CPG_transfer(a, self.CPG_controller )
        self.do_simulation(action, self.frame_skip)
        # print("sensor :", self.sim.data.sensordata[np.arange(39, step=3)])
        # print("force :", self.sim.data.actuator_force)
        # print("length :", self.sim.data.actuator_length)
        # get robot state
        #self._robot_state = self._get_robot_state()
        obs = self._get_obs()
        # print("step {}, next obs : position :{}, velocity:{} joint pos :{}, joint vel:{}".format(self._t_step, self.root_position,
        #                                                              self.root_velocity, self.joint_position, self.joint_velocity))
        if self.trajectory_length>0:
            self.history_buffer.push(self.obs_robot_state)

        velocity_base = self.root_velocity
        reward, other_rewards = self.compute_reward(velocity_base, v_commdand, action, obs)

        # confirm if done
        # q_state = self.state_vector()
        # notdone = np.isfinite(q_state).all() \
        #           and self.get_body_com("torso")[2] >= 0.1 and self.get_body_com("torso")[2] <= 0.6
        # done = not notdone

        done, terminal_reward = self._terminal()

        self._last_root_position = self.root_position
        self._last_root_euler = self.root_euler

        if self.isAddDisturbance:
            self.addDisturbance()

        if self.isRenderTrajectory:
            self.RenderTrajectory()

        if self.isRenderDir:
            cmd_direction = np.array([np.cos(self.current_command[2]), np.sin(self.current_command[2]), 0])

            pos0 = self.root_position + cmd_direction * 0.2
            pos1 = pos0 + cmd_direction * (v_commdand[0] / 0.2)

            self.model.site_pos[0] = pos0
            self.model.site_pos[1] = pos1

            self.model.site_rgba[0] = [1, 0.5, 0, 1]

            self.model.site_rgba[1] = [1, 0, 0, 1]


        self._t_step += 1
        return obs, reward+terminal_reward, done, dict(
            root_position = self.root_position,
            root_euler=self.root_euler,
            velocity_base=velocity_base,
            commands=v_commdand,
            rewards=other_rewards,
            obs = obs,
            motor= action,
            torque = self.sim.data.sensordata[np.arange(39, step=3)],
            root_quat = self.root_quat,
        )

    def _terminal(self):

        q_state = self.state_vector()
        notdone = np.isfinite(q_state).all() \
                  and self.get_body_com("torso")[2] >= 0.1 and self.get_body_com("torso")[2] <= 0.6
        done = not notdone

        if done:
            return True , -1000
        if self._t_step > self.max_steps:
            return True ,0

        return False, 0

    def _get_robot_state(self):
        joint_position = state_M.dot(self.sim.data.qpos[7:7+13].reshape((-1, 1))).flatten()
        joint_velocity = state_M.dot(self.sim.data.qvel[6:6+13].reshape((-1, 1))).flatten()

        root_position = self.get_body_com("torso").flatten()


        rot = np.array(
            [[0., 0., -1., 0.],
             [1., 0., 0., 0.],
             [0., -1., 0., 0.],
             [0., 0., 0., 1.]]
        )
        self._root_rotation = transformations.quaternion_matrix(self.root_quat).dot(rot)
        root_euler = np.array(transformations.euler_from_matrix(self._root_rotation, axes='sxyz'))

        self._root_rotation = self._root_rotation[:3,:3]
        self._inv_root_rotation = self._root_rotation.transpose()#np.linalg.inv(self._root_rotation)[:3,:3]


        # root_euler2 = self.get_orien()
        #
        # print('new euler :', root_euler)
        # print('old euler :', root_euler2)


        #root_euler = self.get_orien()
        #print('old root euler :', root_euler)


        #print('new root euler :', new_euler)


        # print('quat', self.root_quat)
        # print(root_euler)
        root_velocity = (root_position - self.last_root_position) / self.dt
        root_angluar_velocity = ( root_euler - self.last_root_euler) / self.dt

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

    def _get_goal_info(self):
        if self.command_mode == 'full':
            cmd = self.current_command[:2]  #x y
        elif self.command_mode == 'error' :
            cmd = self.current_command[:2] - self.root_velocity[:2] #x y
        elif self.command_mode == 'FandE':
            cmd = np.concatenate([self.current_command[:2],
                                  self.current_command[:2] - self.root_velocity[:2]  # x y
                                  ])
        elif self.command_mode =='dir_vel':

            cmd_direction = np.array([np.cos(self.current_command[2]), np.sin(self.current_command[2]), 0])
            cmd = np.concatenate([self.current_command[0:0] - np.linalg.norm(self.root_velocity[:2]),
                                  cmd_direction,
                                  self.root_direction
                                  ])
        elif self.command_mode =='conv_error':

            conv_error = self.root_inv_rotation.dot(self.current_command  - self.root_velocity )
            cmd = np.concatenate([conv_error[:2],
                                  self.current_command[:2]
                                  ])

        elif self.command_mode == 'no':
            cmd =None
        else:
            raise NotImplementedError
        return cmd


    def _get_obs(self):
        # get robot state
        if self.isRootposNotInObs:
            self.obs_robot_state = self._robot_state[3:]
        else:
            self.obs_robot_state = self._robot_state

        # concat history state
        if self.trajectory_length > 0:
            obs = np.concatenate([
                self.obs_robot_state,
                self.histroy_trajectory.flatten()
            ])
        else:
            obs = self.obs_robot_state

        # concat cmd
        cmd = self._get_goal_info()
        if cmd is not None:
            obs = np.concatenate([obs,
                                  cmd])

        return obs

    def _reset_robot_position(self):
        if self.is_hard_reset_position:

            hard_position = os.getenv('XYZ')
            if hard_position is not None:
                x = hard_position.split('_')[0]
                y = hard_position.split('_')[1]
                z = hard_position.split('_')[2]

            goal_points = [float(x), float(y), float(z)]

            self.init_qpos[:3] = goal_points

            T = np.ones(self.model.nq)
            T[2] = 0

         #    qpos = [-0.29627853,  0.08375629,  0.22849016, -0.11638895 , 0.08747848 , 0.06061479,
         #             -0.04676204, -0.20625531 , 0.50924755, -0.64953141, -0.00216199 ,-0.3749116,
         #              0.12870825]
         #    qvel = [-1.18470342 , 2.58112803 , 1.97923591 , 0.56271407 ,-0.34421984 , 2.73444493,
         # -2.8795056 ,  2.93954109, -1.16187402 , 2.28738732 ,-0.10565884 , 3.11114063,
         #  3.20108197]
            #
            # qpos= [0.08696509 ,-0.02266381, -0.00050443,  0.00114894, -0.05612997 , 0.04914332,
            #        -0.00165076 , 0.20790401, -0.05150042 , 0.44837893 ,-0.02797514 , 0.28012856,
            #        0.06318285]
            # qvel= [1.83018464 , 0.19192611 ,-0.03096657, -0.05215018 , 0.08410683 ,-0.16124939,
            #        0.10221479, -2.90619531, -1.93506711, -2.66933662, -0.27239235, -2.97988054,
            #        -1.88363454]

            self.init_qpos[7:] = qpos
            self.init_qvel[6:] = qvel

            qpos = self.init_qpos  # + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) * T
            qvel = self.init_qvel  # + self.np_random.randn(self.model.nv) * .1


           # print('hard reset : ', qpos)

        else:
            T = np.ones(self.model.nq)
            T[2] = 0

            if self.rand_init == 0:
                qpos = self.init_qpos  # + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) * T
                qvel = self.init_qvel  # + self.np_random.randn(self.model.nv) * .1
            else:
                qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) * T
                qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1

        # for star
        qpos[0:2] = self.command[0,:2]

        # for eight
        #qpos[3:7] =[ 0.65328148,  0.27059805,  0.65328148, -0.27059805]

        # # for heart
        # qpos[0:2] = [0, 0.7]
        # qpos[3:7] =[0.70020647, 0.09854386, 0.70020647, -0.09854386]

        # for rect
        #qpos[0:2] = [0, 0.]


        self.set_state(qpos, qvel)
        self.goal_theta = pi / 4.0
        #self.model.site_pos[1] = [cos(self.goal_theta), sin(self.goal_theta), 0]

        # init positon and euler
        self._last_root_position = self.get_body_com("torso").flatten()
        self._last_root_euler = self.get_orien()

    def _render_goal_position(self, position):
        """
        :param position: [x,y,z]
        :return:
        """
        self.model.site_pos[1] = position


    def _sample_command(self, command):
        if command is  None:
            self.command = command_generator(self.command_max_step , self.dt, self.command_duration,
                                             vx_range=(self.command_vx_low, self.command_vx_high),
                                             vy_range=(self.command_vy_low, self.command_vy_high),
                                             wyaw_range=(self.command_wz_low, self.command_wz_high), render=False)
        else:
            self.command = command

        global_command = os.getenv('GLOBAL_CMD')
        if global_command is not None:
            self.command = IO('{}/{}.pkl'.format(proj_dir, global_command)).read_pickle()
            print('Global command is selected, cmd_{}'.format(global_command))

       # plot_command(self.command)

    def reset_model(self, command = None,  ):


        # reset init robot
        self._reset_robot_position()

        self.goal_orien_yaw =  0
        self.goal_x = 0
        self.goal_y = 0
        self.goal_orien_yaw = 0
        self.goal_xyyaw = np.array([0.0, 0.0, 0.0])

        for i in range(6):
            self.filter_fun_list[i].reset()

        self._sample_command(command)

        # reset something...
        if self.trajectory_length >0:
            self.history_buffer = TrajectoryBuffer(num_size_per=self.robot_state_dim, max_trajectory=self.trajectory_length)

        # set curriculum params...
        self.k0 = set_curriculum(self.curriculum_init)
        self.kd = self.kd_init
        if get_iter_rl() == self.cur_itr_rl:
            self.kc = self.kc
        elif get_iter_rl() == self.cur_itr_rl +1 :
            self.kc = math.pow(self.kc, self.kd)
            self.cur_itr_rl += 1
        else:
            raise Exception('currrent itr seems incorrect')

        obs = self._get_obs()

        if self.trajectory_length>0:
            self.history_buffer.full_state(self.obs_robot_state)
        self._t_step = 0

        # reset something...
        self.CPG_controller = CPG_controller_fun(CPG_node_num=self.num_joint, position_vector=position_vector, dt=self.dt,
                                                 mode=self.cpg_mode)
        self._last_root_position = self.root_position
        self._last_root_euler = self.root_euler

        self._reset_cnt += 1
        return obs

    def viewer_setup(self):
        self.viewer.cam.distance = 6.6#self.model.stat.extent * 0.5
       # self.viewer.cam.lookat = np.array([0,0.00112246279, 0.0820925741])

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def compute_reward(self, velocity_base, v_commdand, action, obs):
        if self.reward_choice == 0:
            ## line
            forward_reward = velocity_base[0]

            ctrl_cost = .5 * np.square(action).sum()
            contact_cost = 0.5 * 1e-4 * np.sum(
                np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 1.0
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward

            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])

        elif self.reward_choice == 1:
            y_pose = self.root_position[1]
            # goal_vel = 0.10

            forward_reward = -1.0 * np.linalg.norm(velocity_base[:2] - v_commdand[:2])
            # y_cost = -0.5 * K_kernel6(y_pose)
            y_cost = -0.5 * np.linalg.norm(y_pose)

            ctrl_cost = 0  # -0.5 * np.square(action).sum()
            contact_cost = -0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 0.0

            reward = forward_reward + ctrl_cost + contact_cost + survive_reward + y_cost
            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, y_cost])

            #print(other_rewards)
        else:
            raise NotImplementedError
        #print(other_rewards)
        return reward, other_rewards


    def CPG_transfer(self, RL_output, CPG_controller ):
        # update the params of CPG
        CPG_controller.update(RL_output)
        #print('action :', RL_output)
        # adjust CPG_neutron parm using RL_output
        output_list = CPG_controller.output(state=RL_output)

        joint_postion_ref = np.array(output_list[1:])
        cur_angles = self.joint_position
        action = position_PID(cur_angles, joint_postion_ref, target_velocities=0)

        return action

    def get_pose(self):
        pos = self.sim.data.qpos[:3]

        q_g2 = self.sim.data.qpos[3:7]

        q = np.array([0.5000, 0.5000, 0.5000, -0.5000])

        R_q = quaternion_multiply(q_g2, quaternion_inverse(q))
        #print(q_g2, q, R_q)

        orien = euler_from_quaternion(R_q, axes='sxyz')

        # # 以上计算效率不一定高
        # Tg2_e = quat2tform(q_g2)
        # Tg1 = Tg1 =
        #
        #     0.0000    1.0000         0         0
        #    -0.0000    0.0000   -1.0000         0
        #    -1.0000    0.0000    0.0000         0
        #          0         0         0    1.0000
        #
        # XYZ = tform2eul(Tg2_e * inv(Tg1), 'XYZ')

        pos = np.concatenate((pos, orien))
        return pos

    def get_orien(self):
        # pos = self.sim.data.qpos[:3]
        q_g2 = self.sim.data.qpos[3:7]
        q = np.array([0.5000, 0.5000, 0.5000, -0.5000])
        R_q = quaternion_multiply(q_g2, quaternion_inverse(q))
        # print(q_g2, q, R_q)
        orien = euler_from_quaternion(R_q, axes='sxyz')
        return np.array(orien)


    @property
    def root_quat(self):
        return self.sim.data.qpos[3:7]
    @property
    def root_rotation(self):
        """
        alion to the globle frame
        :return:
        """
        return  self._root_rotation
        # q = np.array([0.5000, 0.5000, 0.5000, -0.5000])
        # R_q = quaternion_multiply(self.root_quat,  quaternion_inverse(q))
        # return transformations.quaternion_matrix(R_q)[:3,:3]

    @property
    def root_inv_rotation(self):
        return self._inv_root_rotation

    @property
    def root_direction(self):
        return np.array([np.cos(self.root_euler[-1]), np.sin(self.root_euler[-1]), 0])

    @property
    def robot_state(self):
        return self._robot_state
    @property
    def root_position(self):
        return self._robot_state[0:3]
    @property
    def root_euler(self):
        return self._robot_state[3:6]
    @property
    def joint_position(self):
        return self._robot_state[6:19]
    @property
    def joint_velocity(self):
        return  self._robot_state[19:32]
    @property
    def root_velocity(self):
        return self._robot_state[32:35]
    @property
    def root_angluar_velocity(self):
        return self._robot_state[35:38]
    # @property
    # def filterd_root_velocity(self):
    #     return self._robot_state[38:41]
    # @property
    # def filterd_root_angluar_velocity(self):
    #     return self._robot_state[41:44]

    @property
    def last_root_position(self):
        return self._last_root_position
    @property
    def last_root_euler(self):
        return self._last_root_euler

    @property
    def histroy_trajectory(self):
        return self.history_buffer.buffer[-self.trajectory_length:]

    @property
    def current_command(self):
        return self.command[self._t_step]


class CellRobotEnvCPG6GoalTraj(CellRobotEnvCPG6Goal):
    def __init__(self,
                 trajectory_length = 40,
                 trajectory_sample_count = 4,
                 **kwargs):

        self.sample_count = trajectory_sample_count
        #trajectory_length  = 40

        os.environ["NUM_BUFFER"] = str(trajectory_length)
        self.sample_interval = int( trajectory_length / self.sample_count)  # 40/4  10


        self._pred_root_position = np.zeros((self.sample_count, 3), dtype=np.float32)

        CellRobotEnvCPG6Goal.__init__(self, **kwargs)

        self.trajectory_length = trajectory_length
        if self.trajectory_length > 0:
            self.history_buffer = TrajectoryBuffer(num_size_per=self.robot_state_dim,
                                                   max_trajectory=self.trajectory_length)

    def _get_obs(self):
        # get robot state
        self._robot_state = self._get_robot_state()
        if self.isRootposNotInObs:
            self.obs_robot_state = self.robot_state[3:]
        else:
            self.obs_robot_state = self.robot_state
        # concat history state
        if self.trajectory_length > 0:
            obs = np.concatenate([
                self.obs_robot_state,
                self.sampled_history_trajectory.flatten()
            ])
        else:
            obs = self.obs_robot_state

        #print("quat :", self.root_quat)
        # concat cmd
        cmd = self._get_goal_info()
        if cmd is not None:
            obs = np.concatenate([obs,
                                  cmd])
        if self.isRootposNotInObs:
            pred_position =  self._get_pred_root_position()
            obs = np.concatenate([obs,
                                  pred_position,
                                  self.root_rotation.flatten()
                                  ])

        else:
            pred_position = self._get_pred_root_position()
            obs = np.concatenate([obs,
                                  pred_position])

        return obs

    def _get_pred_root_position(self):

        if self.isRootposNotInObs:

            self._pred_root_position[:, 0] = np.array(
                [  self.current_command[0] * self.sample_interval * self.dt * (i + 1) for i in
                 range(self.sample_count)])
            self._pred_root_position[:, 1] = np.array(
                [ self.current_command[1] * self.sample_interval * self.dt * (i + 1) for i in
                 range(self.sample_count)])

            #xyz_pos = self._pred_root_position - self.root_position

            # x_pos = np.array(  [self.root_position[0] + self.current_command[0] * self.sample_interval * self.dt * (i + 1) for i in range(self.sample_count)])
            # y_pos = np.array(  [self.root_position[1] + self.current_command[1] * self.sample_interval * self.dt * (i + 1) for i in range(self.sample_count)])
            # xyz_pos = np.concatenate((x_pos[None], y_pos[None], np.zeros_like(x_pos)[None]), axis=0).transpose() - self.root_position

            xyz_pos = np.array([self.root_inv_rotation.dot(self._pred_root_position[_]) for _ in range(self.sample_count)])
            self._pred_root_position = xyz_pos
            #self._pred_root_position = xyz_pos
        else:
            x_pos = np.array([self.root_position[0] + self.current_command[0] * self.sample_interval * self.dt * (i + 1) for i in range(self.sample_count)])
            y_pos = np.array([self.root_position[1] + self.current_command[1] * self.sample_interval * self.dt * (i + 1) for i in range(self.sample_count)])

            self._pred_root_position[:,0] = x_pos
            self._pred_root_position[:,1] = y_pos
        return self._pred_root_position[:, :2].flatten()


    @property
    def sampled_history_trajectory(self):
        sampled_index =np.arange(0, self.trajectory_length, step=self.sample_interval)
        return self.history_buffer.buffer[sampled_index]

    @property
    def pred_root_positions(self):
        return self._pred_root_position

    @property
    def future_root_position(self):
        return np.array([self.goal_x, self.goal_y])

    def compute_reward(self, velocity_base, v_commdand, action, obs):

        if self.reward_choice == 0:
            ## line
            forward_reward = velocity_base[0]

            ctrl_cost = .5 * np.square(action).sum()
            contact_cost = 0.5 * 1e-4 * np.sum(
                np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 1.0
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward

            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])

        elif self.reward_choice == 1:
            y_pose = self.root_position[1]
            # goal_vel = 0.10

            forward_reward = -1.0 * abs(velocity_base[0] - v_commdand[0])-1.0 * abs(velocity_base[1] - v_commdand[1])

            direction_reward = -0.5 *abs(self.root_euler[2] - v_commdand[2])


            y_cost = -0.5 * np.linalg.norm(self.pred_root_positions - self.future_root_position)

            ctrl_cost = 0  # -0.5 * np.square(action).sum()
            contact_cost = -0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 0.0

            reward = forward_reward + ctrl_cost + contact_cost + survive_reward + y_cost
            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, y_cost, direction_reward])

          #  print(other_rewards)
        elif self.reward_choice == 2:
            """
            x direction velocity tracking
            """
            y_pose = self.root_position[1]
            # goal_vel = 0.10

            forward_reward = -1.0 * np.linalg.norm(velocity_base[:2] - v_commdand[:2])
            # y_cost = -0.5 * K_kernel6(y_pose)
            y_cost = -0.1 * np.linalg.norm(y_pose)

            ctrl_cost = 0  # -0.5 * np.square(action).sum()
            contact_cost = -0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 0.0

            reward = forward_reward + ctrl_cost + contact_cost + survive_reward + y_cost
            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, y_cost])

            #print("test",other_rewards)
        elif self.reward_choice == 3:
            """
            x, y direction velocity tracking
            """

            #y_pose = self.root_position[1]
            # goal_vel = 0.10

            forward_reward = -1.0 * np.linalg.norm(velocity_base[:2] - v_commdand[:2])
            # y_cost = -0.5 * K_kernel6(y_pose)
            y_cost =0 # -0.1 * np.linalg.norm(y_pose)

            ctrl_cost = 0  # -0.5 * np.square(action).sum()
            contact_cost = 0#-0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 0.0

            reward = forward_reward + ctrl_cost + contact_cost + survive_reward + y_cost
            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, y_cost])
            #print('time: {}, reward: {}'.format(self._t_step, other_rewards))
        elif self.reward_choice == 4:
            """
            yaw direction velocity tracking
            """

            #y_pose = self.root_position[1]

            cmd_direction = np.array([np.cos(v_commdand[2]), np.sin(v_commdand[2]), 0])

            forward_reward = -20.0 * abs(np.linalg.norm(velocity_base[:2]) - v_commdand[0])

            direction_reward =  np.inner(cmd_direction, self.root_direction)

            #print('angle: ', np.degrees(np.arccos(direction_reward)))

            forward_reward = np.exp(forward_reward)
            direction_reward = np.exp(direction_reward - 1)


            ctrl_cost = 0  # -0.5 * np.square(action).sum()
            contact_cost = 0#-0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 0.0

            reward = forward_reward * direction_reward + contact_cost - 1
            other_rewards = np.array([reward, forward_reward,direction_reward ])

            #print(other_rewards)

        elif self.reward_choice == 5:
            """
            x, y direction velocity tracking, considering positions
            """


            # goal_vel = 0.10

            forward_reward = -1.0 * np.linalg.norm(velocity_base[:2] - v_commdand[:2])
            position_reward =  -1.0 * np.linalg.norm(np.array([self.goal_x, self.goal_y]) - self.root_position[:2])


            #print("reset cnt :", self._reset_cnt)

            # y_cost = -0.5 * K_kernel6(y_pose)
            y_cost =0 # -0.1 * np.linalg.norm(y_pose)

            ctrl_cost = 0  # -0.5 * np.square(action).sum()
            contact_cost = -0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 0.0

            reward = forward_reward + 0.1* position_reward + ctrl_cost + contact_cost + survive_reward + y_cost
            other_rewards = np.array([reward, forward_reward, position_reward, ctrl_cost, contact_cost, y_cost])

            #print(other_rewards)
            #1- np.exp(-0.01*x)

        elif self.reward_choice == 6:
            """
            x, y direction velocity tracking, considering positions
            """


            # goal_vel = 0.10

            forward_reward = -1.0 * np.linalg.norm(velocity_base[:2] - v_commdand[:2])
            position_reward =  -1.0 * np.linalg.norm(np.array([self.goal_x, self.goal_y]) - self.root_position[:2])


            kc = 1- np.exp(-0.02*self._reset_cnt)

            # y_cost = -0.5 * K_kernel6(y_pose)
            y_cost =0 # -0.1 * np.linalg.norm(y_pose)

            ctrl_cost = 0  # -0.5 * np.square(action).sum()
            contact_cost = -0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 0.0

            reward = forward_reward + kc* 0.2* position_reward + ctrl_cost + contact_cost + survive_reward + y_cost
            other_rewards = np.array([reward, forward_reward, position_reward, ctrl_cost, contact_cost, y_cost])
            # print("reset cnt :", self._reset_cnt)
            # print("kc :", kc)
            # print(other_rewards)
        else:
            raise NotImplementedError
        #print(other_rewards)
        return reward, other_rewards




class CellRobotEnvCPG6NewObsGoalTraj(CellRobotEnvCPG6GoalTraj):
    def __init__(self,

                 **kwargs):

        CellRobotEnvCPG6GoalTraj.__init__(self, **kwargs)

    def _get_obs(self):
        # get robot state
        self._robot_state = self._get_robot_state()
        self.obs_robot_state = self.robot_state

        self.obs_robot_state = np.concatenate([
            self.obs_robot_state,
            self.root_quat.flatten()
        ])


        # concat history state
        if self.trajectory_length > 0:
            history_traj = self.sampled_history_trajectory

            delta_position = history_traj[:, :3] - self.root_position
            root_pos_history = np.array(
                [self.root_inv_rotation.dot(delta_position[_]) for _ in range(delta_position.shape[0])])

            obs = np.concatenate([
                self.obs_robot_state[3:],
                root_pos_history.flatten(),
                history_traj[:, 3:].flatten()

            ])
        else:
            obs = self.obs_robot_state

        #print("quat :", self.root_quat)
        # concat cmd
        cmd = self._get_goal_info()
        if cmd is not None:
            obs = np.concatenate([obs,
                                  cmd])

        pred_position =  self._get_pred_root_position()
        obs = np.concatenate([obs,
                              pred_position,

                              ])


        return obs




