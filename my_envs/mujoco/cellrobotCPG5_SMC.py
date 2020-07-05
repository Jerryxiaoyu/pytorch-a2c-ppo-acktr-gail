import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env
from math import pi, sin, cos
from transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion
from CPG_controllers.controllers.CPG_controller_quadruped import CPG_network_Sinusoid
from CPG_controllers.CPG_process  import  position_PID
from my_envs.base.ExperienceDataset import DataBuffer
from my_envs.base.command_generator import command_generator
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

class CellRobotEnvCPG5(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    This is a new env for SMC journal
    """
    def __init__(self,
                 control_skip = 1,
                 cpg_mode = 0):
        print('test cpg2........................')
        self.reward_choice = os.getenv('REWARD_CHOICE')
        self.cpg_mode = cpg_mode

        self.num_buffer = os.getenv('NUM_BUFFER')
        self.command_mode = os.getenv('COMMAND_MODE')
        self.buffer_mode = os.getenv('BUFFER_MODE')
        self.CPG_enable = os.getenv('CPG_ENABLE')
        self.action_dim = os.getenv('ACTION_DIM')
        self.state_mode = os.getenv('STATE_MODE')
        self.command_vx_high = os.getenv('COMMAND_X')
        self.command_vy_high = os.getenv('COMMAND_Y')
        self.command_wz_high = os.getenv('COMMAND_Z')
        self.vel_filtered = os.getenv('VEL_FILTER')
        self.turing_flag = os.getenv('TURING_FLAG')

        self.xml_name = os.getenv('XML_NAME')

        if self.buffer_mode is None:
            self.buffer_mode = 1
            print('buffer_mode is not sepecified, so buffer_mode is default  1.')
        else:
            self.buffer_mode = int(self.buffer_mode)
            print('buffer_mode = ',self.buffer_mode )

        if self.num_buffer is None:
            self.num_buffer = 1
            print('num_buffer is not sepecified, so num_buffer is default  1.')
        else:
            self.num_buffer = int(self.num_buffer)
            print('num_buffer = ',self.num_buffer )

        if self.command_mode is None:
            self.command_mode = 'delta'
            print('command_mode is not sepecified, so command_mode is default  delta.')
        else:
            #self.command_mode = int(self.command_mode)
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

        if self.state_mode is None:
            self.state_mode = 'pos'
            print('STATE_MODE is not sepecified, so state_mode is default .')
        else:

            print('STATE_MODE = ', self.state_mode)

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
        if self.command_wz_high is None:
            self.command_wz_high = 0
            print('command_wz_high is not sepecified, so command_wz_high = 0 .')
        else:

            print('command_wz_high = ', self.command_wz_high)


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
            self.xml_name = 'cellrobot_Quadruped_float.xml'

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
        #self.buffer_mode = 1
        #self.num_buffer = 2
        #self.command_mode = 'delta'  # 'command' None, 'delta'

        self.num_joint = 13
        policy_a_dim = 13  # networt output
        self.command = command_generator(10000, 0.01, 2, render = False)
        self.c_index = 0
        self.c_index_max = 10000
        self.action_pre = 0
        dt = 0.01
        self.goal_orien_yaw = 0
        self.goal_x = 0
        self.goal_y = 0
        self.goal_xyyaw = np.array([0.0, 0.0,0.0])

        self.command_vx_low = 0
        self.command_vx_high = float(self.command_vx_high)
        self.command_vy_low = 0
        self.command_vy_high = float(self.command_vy_high)
        self.command_wz_low =0
        self.command_wz_high = float(self.command_wz_high)

        self.command_max_step = 10000  # steps
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

        self.Calc_Reward = self.reward_fun1
        self.CPG_controller = CPG_controller_fun(CPG_node_num=self.num_joint,
                                                 position_vector=position_vector, dt=dt,
                                                 mode = self.cpg_mode)

        if self.buffer_mode == 1:
            self.size_buffer_data = self.num_joint * 2 + policy_a_dim
        elif self.buffer_mode == 2:
            self.size_buffer_data = self.num_joint * 2 + policy_a_dim + 6
        elif self.buffer_mode == 3:
            self.size_buffer_data = self.num_joint * 2
        elif self.buffer_mode == 4:
            self.size_buffer_data = self.num_joint * 2 + 6
        elif self.buffer_mode == 5:
            self.size_buffer_data = 2
        else:
            raise Exception("buffer_mode is not correct!")

        self.history_buffer = DataBuffer(num_size_per=self.size_buffer_data, max_trajectory=self.num_buffer)

        self.vel_num_buffer = 100
        self.vel_buffer = DataBuffer(num_size_per= 3, max_trajectory= self.vel_num_buffer)

        cutoff_hz = 10 #Hz
        self.reward_fir= fir_filter(int(1/dt),cutoff_hz,10)

        self.fir_vx = fir_filter(100, 0.01, 30)
        self.fir_vy = fir_filter(100, 0.1, 30)
        self.fir_wz = fir_filter(100, 0.1, 30)

        # old path :'cellrobot/cellrobot_Quadruped_float_simple.xml

        self.control_skip = control_skip

        mujoco_env.MujocoEnv.__init__(self, self.model_path,  self.control_skip , custom_action_space= self.custom_action_space)  # cellrobot_test_gen  CR_quadruped_v1_A001  'cellrobot/cellrobot_test_gen.xml' Atlas_v5/atlas_v5.xml
        utils.EzPickle.__init__(self)
        seed = mujoco_env.MujocoEnv.seed(self)

        print('seed = ', seed)
        np.random.seed(seed[0]>>32)
        print('1111REWARD_CHOICE = ', os.getenv('REWARD_CHOICE'))
        #seeding.np_random(seed[0])

        print('State size :', self.observation_space.shape[0])
        print('Policy action size : ', self.action_space.shape[0] )

    def step(self, a):
        action = a.copy()

        v_commdand = self.command[self.c_index, :3]
        self.goal_orien_yaw += v_commdand[-1]*self.dt
        # self.goal_x += v_commdand[0]*self.dt
        # self.goal_y += v_commdand[1]*self.dt
        self.goal_xyyaw += v_commdand*self.dt

        pose_pre = np.concatenate((self.get_body_com("torso"), self.get_orien()))

        obs = self._get_obs()

        if self.CPG_enable == 1 :
            action = CPG_transfer(a, self.CPG_controller, obs)
            #print(action)


        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        pose_post = obs[:6]
        velocity_base = (pose_post - pose_pre) / self.dt  # dt 0.01

        # xposbefore = self.get_body_com("torso")[0]
        #
        # xposafter = self.get_body_com("torso")[0]
        # forward_reward = (xposafter - xposbefore) / self.dt

        vx_f = self.fir_vx.apply(velocity_base[0])
        vy_f = self.fir_vy.apply(velocity_base[1])
        wz_f = self.fir_wz.apply(velocity_base[-1])
        vel_f = np.concatenate((np.array([vx_f]), np.array([vy_f]), velocity_base[2:5],  np.array([wz_f])))


        state = self.state_concatenate(obs, pose_pre, self.history_buffer, self.command[self.c_index], vel_f,
                                       num_buffer=self.num_buffer)


        if self.buffer_mode == 1: # joint and action
            toStoreData = np.concatenate((obs[6:32], action), axis=0)
        elif self.buffer_mode ==2:
            toStoreData = np.concatenate((obs[0:32], action), axis=0)
        elif self.buffer_mode ==3:  # only joint
            toStoreData = np.concatenate((obs[6:32] ), axis=0)
        elif self.buffer_mode == 4:
            toStoreData = np.concatenate((obs[0:32]), axis=0)
        elif self.buffer_mode == 5:
            toStoreData = a
        else:
            raise Exception("buffer_mode is not correct!")

        if self.vel_filtered:
            velocity_base =vel_f

        reward, other_rewards = self.Calc_Reward( velocity_base, v_commdand, action, obs)

        self.action_pre = action
        self.c_index += 1

        # confirm if done
        q_state = self.state_vector()
        notdone = np.isfinite(q_state).all() \
                  and self.get_body_com("torso")[2] >= 0.1 and self.get_body_com("torso")[2] <= 0.6
        done = not notdone

        return state, reward, done, dict(
            velocity_base=velocity_base,
            commands=v_commdand,
            rewards=other_rewards,
            obs = obs
        )

    def _get_obs(self):
        orien = self.get_orien()

        obs = np.concatenate([
            self.get_body_com("torso").flat,  # base x, y, z  0-3
            orien,  # oren   3-6
            state_M.dot(self.sim.data.qpos[7:].reshape((-1, 1))).flat,  # joint positon  6-19
            state_M.dot(self.sim.data.qvel[6:].reshape((-1, 1))).flat  # joint velosity 19-32
        ])

        return obs

    def reset_model(self, command = None, reward_fun_choice = None):
        T = np.ones(self.model.nq)
        T[2] = 0


        if self.rand_init == 0:
            qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) * T
            qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
        else:
            qpos = self.init_qpos +  self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) * T
            qvel = self.init_qvel +   self.np_random.randn(self.model.nv) * .1

        self.set_state(qpos, qvel)
        self.goal_theta = pi / 4.0
        self.model.site_pos[1] = [cos(self.goal_theta), sin(self.goal_theta), 0]


        self.goal_orien_yaw =  0
        self.goal_x = 0
        self.goal_y = 0

        self.fir_vx.reset()
        self.fir_vy.reset()
        self.fir_wz.reset()

        if command is  None:
            self.command = command_generator(self.command_max_step , self.dt, self.command_duration,
                                             vx_range=(self.command_vx_low, self.command_vx_high),
                                             vy_range=(self.command_vy_low, self.command_vy_high),
                                             wyaw_range=(self.command_wz_low, self.command_wz_high), render=False)
        else:
            self.command = command

        global_command = os.getenv('GLOBAL_CMD')
        if global_command is not None:
            self.command = IO('data/cmd_{}.pkl'.format(global_command)).read_pickle()
            print('Global command is selected, cmd_{}'.format(global_command))

        #global reward_choice
        if self.reward_choice is None:
            print('REWARD_CHOICE is not specified!')
            reward_fun_choice_env =1
        else:
            reward_fun_choice_env = int(self.reward_choice)
           # print('REWARD_CHOICE is ', reward_fun_choice_env)
        # print('2222REWARD_CHOICE = ', os.getenv('REWARD_CHOICE'))
        if reward_fun_choice is None:
            reward_fun_choice = reward_fun_choice_env

        # print('reward_fun_choice = ', reward_fun_choice)
        # print('reward_fun_choice_env = ', reward_fun_choice_env)
        if reward_fun_choice == 1:
            self.Calc_Reward = self.reward_fun1
        elif reward_fun_choice == 2:
            self.Calc_Reward = self.reward_fun2
        elif reward_fun_choice == 3:
            self.Calc_Reward = self.reward_fun3
        elif reward_fun_choice == 4:
            self.Calc_Reward = self.reward_fun4
        elif reward_fun_choice == 5:
            self.Calc_Reward = self.reward_fun5
        elif reward_fun_choice == 6:
            self.Calc_Reward = self.reward_fun6
        elif reward_fun_choice == 7:
            self.Calc_Reward = self.reward_fun7
        elif reward_fun_choice == 8:
            self.Calc_Reward = self.reward_fun8
        elif reward_fun_choice == 9:
            self.Calc_Reward = self.reward_fun9
        elif reward_fun_choice == 10:
            self.Calc_Reward = self.reward_fun10
        elif reward_fun_choice == 11:
            self.Calc_Reward = self.reward_fun11
        elif reward_fun_choice == 12:
            self.Calc_Reward = self.reward_fun12
        elif reward_fun_choice == 13:
            self.Calc_Reward = self.reward_fun13
        elif reward_fun_choice == 14:
            self.Calc_Reward = self.reward_fun14
        elif reward_fun_choice == 15:
            self.Calc_Reward = self.reward_fun15
        elif reward_fun_choice == 16:
            self.Calc_Reward = self.reward_fun16
        elif reward_fun_choice == 17:
            self.Calc_Reward = self.reward_fun17
        elif reward_fun_choice == 18:
            self.Calc_Reward = self.reward_fun18
        elif reward_fun_choice == 19:
            self.Calc_Reward = self.reward_fun19
        elif reward_fun_choice == 20:
            self.Calc_Reward = self.reward_fun20
        elif reward_fun_choice == 21:
            self.Calc_Reward = self.reward_fun21
        elif reward_fun_choice == 22:
            self.Calc_Reward = self.reward_fun22
        elif reward_fun_choice == 23:
            self.Calc_Reward = self.reward_fun23

        elif reward_fun_choice == 24:
            self.Calc_Reward = self.reward_fun24
        elif reward_fun_choice == 25:
            self.Calc_Reward = self.reward_fun25
        elif reward_fun_choice == 26:
            self.Calc_Reward = self.reward_fun26
        elif reward_fun_choice == 27:
            self.Calc_Reward = self.reward_fun27
        elif reward_fun_choice == 28:
            self.Calc_Reward = self.reward_fun28
        elif reward_fun_choice == 29:
            self.Calc_Reward = self.reward_fun29
        elif reward_fun_choice == 30:
            self.Calc_Reward = self.reward_fun30
        elif reward_fun_choice == 31:
            self.Calc_Reward = self.reward_fun31
        elif reward_fun_choice == 32:
            self.Calc_Reward = self.reward_fun32
        elif reward_fun_choice == 33:
            self.Calc_Reward = self.reward_fun33
        elif reward_fun_choice == 34:
            self.Calc_Reward = self.reward_fun34
        elif reward_fun_choice == 35:
            self.Calc_Reward = self.reward_fun35
        elif reward_fun_choice == 36:
            self.Calc_Reward = self.reward_fun36
        elif reward_fun_choice == 37:
            self.Calc_Reward = self.reward_fun37
        elif reward_fun_choice == 38:
            self.Calc_Reward = self.reward_fun38
        elif reward_fun_choice == 39:
            self.Calc_Reward = self.reward_fun39
        elif reward_fun_choice == 40:
            self.Calc_Reward = self.reward_fun40
        elif reward_fun_choice == 41:
            self.Calc_Reward = self.reward_fun41
        elif reward_fun_choice == 42:
            self.Calc_Reward = self.reward_fun42
        elif reward_fun_choice == 43:
            self.Calc_Reward = self.reward_fun43
        elif reward_fun_choice == 44:
            self.Calc_Reward = self.reward_fun44
        elif reward_fun_choice is None:
            self.Calc_Reward = self.reward_fun1
            reward_fun_choice = 1
        else:
            raise Exception('reward fun error!')

        self.goal_orien_yaw = 0
        self.goal_xyyaw = np.array([0.0, 0.0, 0.0])
        #print('Reward function: ', reward_fun_choice)
        self.c_index = 0
        self.history_buffer = DataBuffer(num_size_per=self.size_buffer_data, max_trajectory=self.num_buffer)
        self.vel_buffer = DataBuffer(num_size_per=3, max_trajectory=self.vel_num_buffer)

        self.k0 = set_curriculum(self.curriculum_init)
        self.kd = self.kd_init
        if get_iter_rl() == self.cur_itr_rl:
            self.kc = self.kc
        elif get_iter_rl() == self.cur_itr_rl +1 :
            self.kc = math.pow(self.kc, self.kd)
            self.cur_itr_rl += 1
        else:
            raise Exception('currrent itr seems incorrect')

        #pre_pose = np.zeros(6)
        obs = self._get_obs()

        pre_pose = obs[:6]
        action = np.zeros(13)
        vel_f = np.zeros(3)

        state = self.state_concatenate(obs, pre_pose, self.history_buffer, self.command[self.c_index],vel_f,
                                       num_buffer=self.num_buffer)

        if self.buffer_mode == 1:
            toStoreData = np.concatenate((obs[6:32], action), axis=0)
        elif self.buffer_mode == 2:
            toStoreData = np.concatenate((obs[0:32], action), axis=0)
        elif self.buffer_mode == 3:
            toStoreData = np.concatenate((obs[6:32]), axis=0)
        elif self.buffer_mode == 4:
            toStoreData = np.concatenate((obs[0:32]), axis=0)
        elif self.buffer_mode == 5:
            toStoreData = np.zeros(2)
        else:
            raise Exception("buffer_mode is not correct!")
        self.history_buffer.push(toStoreData)


        self.CPG_controller = CPG_controller_fun(CPG_node_num=self.num_joint, position_vector=position_vector, dt=self.dt)
        return state

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def get_pose(self):
        pos = self.sim.data.qpos[:3]

        q_g2 = self.sim.data.qpos[3:7]

        q = np.array([0.5000, 0.5000, 0.5000, -0.5000])

        R_q = quaternion_multiply(q_g2, quaternion_inverse(q))
        print(q_g2, q, R_q)

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

        return orien

    def state_concatenate(self, obs, pose_pre, history_buffer, command, vel_f, num_buffer=2, ):
        """

        :param obs:
        :param history_buffer:
        :param command:
        :return:
        """

        velocity_base = (obs[:6] - pose_pre) / self.dt  # dt 0.01

        data_tmp = history_buffer.pull().copy()[::-1]  # reverse output
        data_size = history_buffer.num_size_per

        if len(data_tmp) == 0:
            data_history = np.zeros(data_size * num_buffer)
        else:
            for i in range(len(data_tmp)):
                if i == 0:
                    data_history = data_tmp[0]
                else:
                    data_history = np.append(data_history, data_tmp[i])
            if len(data_tmp) < num_buffer:
                for i in range(num_buffer - len(data_tmp)):
                    data_history = np.append(data_history, np.zeros(data_size))

        if self.state_mode == 'vel':
            vel = np.concatenate((velocity_base[:2], velocity_base[-1:]))
            state = np.concatenate((vel, obs[2:]))
        elif self.state_mode == 'vel_f':
            vel = np.concatenate((vel_f[:2], vel_f[-1:]))
            state = np.concatenate((vel, obs[2:]))
        elif self.state_mode == 'pos':
            state = obs
        else:
            raise  Exception('state mode is not corect!')
        if num_buffer > 0:
            state = np.append(state, data_history.reshape((1, -1)))

        if self.command_mode == 'command' or self.command_mode == 'error':
            vel = (np.concatenate((obs[:2], obs[5:6])) - np.concatenate((pose_pre[:2], pose_pre[5:6]))) / self.dt
            v_e = vel - command

            state = np.append(state, v_e)
        elif self.command_mode == 'delta'or self.command_mode == 'full':

            state = np.append(state, command)
        elif self.command_mode == 'no':
            state = state
        else:
            raise  Exception(' command mode is not corect!')

        return state

    def reward_fun1(self, velocity_base, v_commdand, action, obs):
            ## line

            forward_reward = velocity_base[0]

            ctrl_cost = .5 * np.square(action).sum()
            contact_cost = 0.5 * 1e-4 * np.sum(
                np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 1.0
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward

            other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])
            #print(other_rewards)
            return reward, other_rewards

    def reward_fun2(self, velocity_base, v_commdand, action, obs):
        rien = obs[3:6]
        x_pose = obs[0]
        y_pose = obs[1]
        #goal_vel = 0.10

        forward_reward = -1.0 * abs(velocity_base[0] - v_commdand[0])
        # y_cost = -0.5 * K_kernel6(y_pose)
        y_cost = -0.5 * np.linalg.norm(y_pose)

        ctrl_cost = 0  # -0.5 * np.square(action).sum()
        contact_cost = -0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0

        reward = forward_reward + ctrl_cost + contact_cost + survive_reward + y_cost
        other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, y_cost])

        return reward, other_rewards





def K_kernel_new2(x):
   # x = np.linalg.norm(x)
    K =  1 / (np.exp(x / 0.02) + 2 + np.exp(-x / 0.02))
    return K

def K_kernel_new(x):
    x = np.abs(x)
    K = 5* 1 / (np.exp(x / 0.08) + 2 + np.exp(-x / 0.08))
    return K


def K_kernel(x):
    x = np.linalg.norm(x)
    K = -1 / (np.exp(x) + 2 + np.exp(-x))
    return K

def K_kernel2(x):
    x = np.linalg.norm(x)
    x = np.clip(x, 0, 3)
    K = -1 / (np.exp(x / 0.2) + np.exp(-x / 0.2))
    return K

def K_kernel3(x):
    x = np.linalg.norm(x)
    K = -1 / (np.exp(x / 0.1) + 2 + np.exp(-x / 0.1))
    return K
def K_kernel4(x):
    x = np.linalg.norm(x)
    K = -1 / (np.exp(x) + 2 + np.exp(-x))
    return K

def K_kernel5(x):
    x = np.linalg.norm(x)
    x = np.clip(x, 0, 3)
    if x > 0.1:
        K = -1 / (np.exp(x / 0.2) + np.exp(-x / 0.2))
    else:
        K = -2 / (np.exp(x / 0.2) + np.exp(-x / 0.2))
    #K = -2 / (np.exp(x / 0.1) + np.exp(-x / 0.1))
    return K
def K_kernel6(x):
    x = np.linalg.norm(x)
    x = np.clip(x, 0, 10)

    K =   -1 / (np.exp(x / 0.5) + 2+ np.exp(-x / 0.5))
    return K
def CPG_transfer(RL_output, CPG_controller, obs):
    # update the params of CPG
    CPG_controller.update(RL_output)

    # adjust CPG_neutron parm using RL_output
    output_list = CPG_controller.output(state=None)

    joint_postion_ref = np.array(output_list[1:])
    cur_angles = obs[obs_low:obs_high]
    action = position_PID(cur_angles, joint_postion_ref, target_velocities=0)

    return action






