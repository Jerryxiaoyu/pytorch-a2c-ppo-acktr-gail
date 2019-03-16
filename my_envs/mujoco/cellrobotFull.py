import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env
from math import pi,sin,cos
from transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion
import os

state_M =np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
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

from my_envs.base.ExperienceDataset import DataBuffer
from my_envs.base.command_generator import command_generator
import time


class CellRobotEnvFull(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        num_joint =13
        policy_a_dim =13  # networt output
        self.command = command_generator(10000, 0.01, 2)
        self.c_index = 0
        self.c_index_max = 10000
        self.action_pre = 0
        dt = 0.01
        self.buffer_mode =1
        self.num_buffer = 0
        self.deta_vel = True
        self.goal_orien_yaw = 0

        self.command_vx_low = 0
        self.command_vx_high = 0.5
        self.command_vy_low = 0
        self.command_vy_high = 0
        self.command_wz_low = 0
        self.command_wz_high = 0

        self.command_max_step = 10000  # steps
        self.command_duration = 2  # second


        self.Calc_Reward = self.reward_fun1

        if self.buffer_mode == 1:
            self.size_buffer_data = num_joint * 2 + policy_a_dim

        elif self.buffer_mode ==2:
            self.size_buffer_data = num_joint * 2 + policy_a_dim +6
        elif self.buffer_mode ==3:
            self.size_buffer_data = num_joint * 2
        elif self.buffer_mode == 4:
            self.size_buffer_data = num_joint * 2 +6
        else:
            raise Exception("buffer_mode is not correct!")


        self.history_buffer = DataBuffer(num_size_per=self.size_buffer_data, max_trajectory=self.num_buffer)

        mujoco_env.MujocoEnv.__init__(self, 'cellrobot/cellrobot_Quadruped_float_simple.xml', 1)  #cellrobot_test_gen  CR_quadruped_v1_A001  'cellrobot/cellrobot_test_gen.xml' Atlas_v5/atlas_v5.xml
        utils.EzPickle.__init__(self)


        
    def step(self, a):
        action = a

        v_commdand = self.command[self.c_index, :3]

        pose_pre = np.concatenate((self.get_body_com("torso"), self.get_orien() ))


        self.do_simulation(action, self.frame_skip)



        obs = self._get_obs()

        state = self.state_concatenate(obs, pose_pre,  self.history_buffer, self.command[self.c_index], num_buffer=self.num_buffer)


        if self.buffer_mode == 1:
            toStoreData = np.concatenate((obs[6:32], action), axis=0)
        elif self.buffer_mode ==2:
            toStoreData = np.concatenate((obs[0:32], action), axis=0)
        elif self.buffer_mode ==3:
            toStoreData = np.concatenate((obs[6:32] ), axis=0)
        elif self.buffer_mode == 4:
            toStoreData = np.concatenate((obs[0:32]), axis=0)
        else:
            raise Exception("buffer_mode is not correct!")
        self.history_buffer.push(toStoreData)



        pose_post = obs[:6]
        velocity_base = (pose_post - pose_pre)/self.dt       #dt 0.01

        reward, other_rewards = self.Calc_Reward(velocity_base, v_commdand, action, obs)

        self.action_pre = action
        self.c_index += 1

        # confirm if done
        q_state = self.state_vector()
        notdone = np.isfinite(q_state).all()  \
                  and state[2] >= 0.1 and state[2] <= 0.6
        done = not notdone


        #print('t1 = {}, t2 ={}, t3 = {}', end_t1, end_t2, end_t3)

        return state, reward, done, dict(
            velocity_base = velocity_base,
            commands = v_commdand,
            rewards = other_rewards,
            obs=obs
            )

    def _get_obs(self):
        orien = self.get_orien()


        obs =np.concatenate([
                        self.get_body_com("torso").flat,   # base x, y, z  0-3
                        orien ,      # oren   3-6
                        state_M.dot(self.sim.data.qpos[7:].reshape((-1, 1))).flat,  # joint positon  6-19
                        state_M.dot(self.sim.data.qvel[6:].reshape((-1, 1))).flat   # joint velosity 19-32
                        ])

        return obs



    def reset_model(self,command = None, reward_fun_choice = None):
        qpos = self.init_qpos  # + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel  # + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.goal_theta = pi / 4.0
        self.model.site_pos[1] = [cos(self.goal_theta), sin(self.goal_theta), 0]


        reward_choice = os.getenv('REWARD_CHOICE')
        if reward_choice is None:
            print('REWARD_CHOICE is not specified!')
            reward_fun_choice_env = 1
        else:
            reward_fun_choice_env = int(reward_choice)

        if command is None:
            self.command = command_generator(self.command_max_step, self.dt, self.command_duration,
                                             vx_range=(self.command_vx_low, self.command_vx_high),
                                             vy_range=(self.command_vy_low, self.command_vy_high),
                                             wyaw_range=(self.command_wz_low, self.command_wz_high))
        else:
            self.command = command

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
        elif reward_fun_choice is None:
            self.Calc_Reward = self.reward_fun1
            reward_fun_choice = 1
        else:
            raise Exception('reward fun error!')

        self.goal_orien_yaw = 0
        # print('Reward function: ', reward_fun_choice)
        self.c_index = 0
        self.history_buffer = DataBuffer(num_size_per=self.size_buffer_data, max_trajectory=self.num_buffer)

        pre_pose = np.zeros(6)
        obs = self._get_obs()
        action = np.zeros(13)

        state = self.state_concatenate(obs, pre_pose, self.history_buffer, self.command[self.c_index], num_buffer=self.num_buffer)

        if self.buffer_mode == 1:
            toStoreData = np.concatenate((obs[6:32], action), axis=0)
        elif self.buffer_mode ==2:
            toStoreData = np.concatenate((obs[0:32], action), axis=0)
        elif self.buffer_mode ==3:
            toStoreData = np.concatenate((obs[6:32] ), axis=0)
        elif self.buffer_mode == 4:
            toStoreData = np.concatenate((obs[0:32]), axis=0)
        else:
            raise Exception("buffer_mode is not correct!")
        self.history_buffer.push(toStoreData)


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
        #pos = self.sim.data.qpos[:3]

        q_g2 = self.sim.data.qpos[3:7]

        q = np.array([0.5000, 0.5000, 0.5000, -0.5000])

        R_q = quaternion_multiply(q_g2, quaternion_inverse(q))
        #print(q_g2, q, R_q)

        orien = euler_from_quaternion(R_q, axes='sxyz')


        return orien

    def state_concatenate(self, obs ,pose_pre,  history_buffer, command, num_buffer=2 ):
        """

        :param obs:
        :param history_buffer:
        :param command:
        :return:
        """

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

        state = obs
        if num_buffer >0:
            state = np.append(state, data_history.reshape((1, -1)))
        

        if self.deta_vel :
            vel = ( np.concatenate((obs[:2], obs[5:6])) - np.concatenate((pose_pre[:2], pose_pre[5:6])) )/self.dt
            v_e = vel - command

            state = np.append(state, v_e)
        else:
            state = np.append(state, command)

        return state

    def reward_fun1(self,velocity_base, v_commdand, action, obs):
        v_e = np.concatenate((velocity_base[:2], velocity_base[-1:])) - v_commdand  # x, y, yaw
        vxy = v_commdand[:2]
        wyaw = v_commdand[2]

        c_f = -1
        c_f2 = -0.2
        forward_reward = c_f * np.linalg.norm(velocity_base[0:2] - vxy) + c_f2 *  np.linalg.norm(velocity_base[2] - wyaw)


        ctrl_cost = -0.005 * np.square(action).sum()
        contact_cost = -0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.5
        reward = forward_reward + ctrl_cost + contact_cost + survive_reward
        other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])

        return reward, other_rewards

    def reward_fun2(self, velocity_base, v_commdand, action, obs):
        #print('reward2')
        v_e = np.concatenate((velocity_base[:2], velocity_base[-1:])) - v_commdand  # x, y, yaw
        vxy = v_commdand[:2]
        wyaw = v_commdand[2]

        c_f = -1
        c_f2 = -0.2
        forward_reward = c_f * K_kernel2(velocity_base[0:2] - vxy) + c_f2 * K_kernel2(velocity_base[2] - wyaw)

        ctrl_cost = -0.005 * np.square(action).sum()
        contact_cost = -0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.2
        reward = forward_reward + ctrl_cost + contact_cost + survive_reward
        other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])

        return reward, other_rewards

    def reward_fun3(self, velocity_base, v_commdand, action, obs):
        # print('reward2')
        v_e = np.concatenate((velocity_base[:2], velocity_base[-1:])) - v_commdand  # x, y, yaw
        vxy = v_commdand[:2]
        wyaw = v_commdand[2]

        c_f = -2
        c_f2 = -0.2
        forward_reward = c_f * K_kernel3(velocity_base[0:2] - vxy) + c_f2 * K_kernel3(velocity_base[2] - wyaw)

        ctrl_cost = -0.005 * np.square(action).sum()
        contact_cost = -0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.2
        reward = forward_reward + ctrl_cost + contact_cost + survive_reward
        other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])

        return reward, other_rewards
    def reward_fun4(self, velocity_base, v_commdand, action, obs):
        # print('reward2')
        v_e = np.concatenate((velocity_base[:2], velocity_base[-1:])) - v_commdand  # x, y, yaw
        vxy = v_commdand[:2]
        wyaw = v_commdand[2]


        c_f = -30 * self.dt
        c_f2 = -6* self.dt
        forward_reward = c_f * K_kernel3(velocity_base[0:2] - vxy) + c_f2 * K_kernel3(velocity_base[2] - wyaw)

        ctrl_cost = -0.005 * np.square(action).sum()
        contact_cost = -0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.2
        reward = forward_reward + ctrl_cost + contact_cost + survive_reward
        other_rewards = np.array([reward, forward_reward, ctrl_cost, contact_cost, survive_reward])

        return reward, other_rewards

    def reward_fun5(self, velocity_base, v_commdand, action, obs):
        v_e = np.concatenate((velocity_base[:2], velocity_base[-1:])) - v_commdand  # x, y, yaw
        vxy = v_commdand[:2]
        wyaw = v_commdand[2]
        q_vel = obs[19:32]
        orien = obs[3:6]

        vx = v_commdand[0]
        vy = v_commdand[1]

        # reward calculate
        kc = 1
        c_w = -2 * self.dt
        c_v1 = -10 * self.dt
        c_v2 = -1 * self.dt
        # lin_vel_cost = c_v1 * K_kernel(c_v2 * (velocity_base[:2] - vxy))
        lin_vel_reward = c_v1 *np.linalg.norm(velocity_base[0:2] - vxy) #np.linalg.norm(velocity_base[0] - vx) + np.linalg.norm(velocity_base[1] - vy)   # c_v1 * (K_kernel3((velocity_base[0] - vx)) + K_kernel3((velocity_base[1] - vy)))
        ang_vel_reward =  c_w * np.linalg.norm(velocity_base[-1] - wyaw)

        c_t = 0.0005 * self.dt
        torque_cost =  -kc * c_t * np.square(action).sum()

        c_js = 0.03 * self.dt
        joint_speed_cost =   -kc * c_js * np.square(q_vel).sum()

        c_0 = 0.4 * self.dt
        orientation_cost = 0
        # orientation_cost = kc * c_0 * np.sqrt([0,0,-1] - orien).sum()  # TODO need to debug , otherwise output nan
        c_s = 0.5 * self.dt
        smoothness_cost =   -kc * c_s * np.square(self.action_pre - action).sum()
        survive_reward = 0.2
        reward = lin_vel_reward + ang_vel_reward + torque_cost + joint_speed_cost + orientation_cost + smoothness_cost + survive_reward

        #reward = self.reward_fir.apply(reward)

        other_rewards = np.array([reward, lin_vel_reward, ang_vel_reward, torque_cost, joint_speed_cost,orientation_cost, smoothness_cost, survive_reward ])

        return reward, other_rewards

    def reward_fun6(self, velocity_base, v_commdand, action, obs):
        '''
        add orien
        :param velocity_base:
        :param v_commdand:
        :param action:
        :param obs:
        :return:
        '''
        v_e = np.concatenate((velocity_base[:2], velocity_base[-1:])) - v_commdand  # x, y, yaw
        vxy = v_commdand[:2]
        wyaw = v_commdand[2]
        q_vel = obs[19:32]
        orien = obs[3:6]

        vx = v_commdand[0]
        vy = v_commdand[1]

        # reward calculate
        kc = 1
        c_w = -2 * self.dt
        c_v1 = -10 * self.dt
        c_v2 = -1 * self.dt
        # lin_vel_cost = c_v1 * K_kernel(c_v2 * (velocity_base[:2] - vxy))
        lin_vel_reward = c_v1 * np.linalg.norm(velocity_base[
                                               0:2] - vxy)  # np.linalg.norm(velocity_base[0] - vx) + np.linalg.norm(velocity_base[1] - vy)   # c_v1 * (K_kernel3((velocity_base[0] - vx)) + K_kernel3((velocity_base[1] - vy)))
        ang_vel_reward = c_w * np.linalg.norm(velocity_base[-1] - wyaw)

        c_t = 0.0005 * self.dt
        torque_cost = -kc * c_t * np.square(action).sum()

        c_js = 0.03 * self.dt
        joint_speed_cost = -kc * c_js * np.square(q_vel).sum()

        c_0 = 0.4 * self.dt
        orientation_cost = 0
        orientation_cost = kc * c_0 * np.square([0,0] - orien[:2]).sum()  # TODO need to debug , otherwise output nan
        c_s = 0.5 * self.dt
        smoothness_cost = -kc * c_s * np.square(self.action_pre - action).sum()
        survive_reward = 0.2
        reward = lin_vel_reward + ang_vel_reward + torque_cost + joint_speed_cost + orientation_cost + smoothness_cost + survive_reward

        # reward = self.reward_fir.apply(reward)

        other_rewards = np.array(
            [reward, lin_vel_reward, ang_vel_reward, torque_cost, joint_speed_cost, orientation_cost, smoothness_cost,
             survive_reward])

        return reward, other_rewards
    def reward_fun7(self, velocity_base, v_commdand, action, obs):
        '''
        integal orien
        :param velocity_base:
        :param v_commdand:
        :param action:
        :param obs:
        :return:
        '''
        v_e = np.concatenate((velocity_base[:2], velocity_base[-1:])) - v_commdand  # x, y, yaw
        vxy = v_commdand[:2]
        wyaw = v_commdand[2]
        q_vel = obs[19:32]
        orien = obs[3:6]

        vx = v_commdand[0]
        vy = v_commdand[1]

        # reward calculate
        kc = 1
        c_w = -2 * self.dt
        c_v1 = -10 * self.dt
        c_v2 = -1 * self.dt
        # lin_vel_cost = c_v1 * K_kernel(c_v2 * (velocity_base[:2] - vxy))
        lin_vel_reward = c_v1 * np.linalg.norm(velocity_base[
                                               0:2] - vxy)  # np.linalg.norm(velocity_base[0] - vx) + np.linalg.norm(velocity_base[1] - vy)   # c_v1 * (K_kernel3((velocity_base[0] - vx)) + K_kernel3((velocity_base[1] - vy)))
        ang_vel_reward = c_w * np.linalg.norm(velocity_base[-1] - wyaw)

        c_t = 0.0005 * self.dt
        torque_cost = -kc * c_t * np.square(action).sum()

        c_js = 0.03 * self.dt
        joint_speed_cost = -kc * c_js * np.square(q_vel).sum()

        c_0 = 0.4 * self.dt
        orientation_cost = 0
        orientation_cost = kc * c_0 * np.square([0, 0] - orien[:2]).sum()  # TODO need to debug , otherwise output nan
        c_s = 0.5 * self.dt
        smoothness_cost = -kc * c_s * np.square(self.action_pre - action).sum()
        survive_reward = 0.2

        c_y = 5 * self.dt
        orien_yaw_cost = - c_y * np.linalg.norm(orien[-1]- self.goal_orien_yaw)
        reward = lin_vel_reward + ang_vel_reward + torque_cost + joint_speed_cost + orientation_cost + smoothness_cost + survive_reward + orien_yaw_cost

        # reward = self.reward_fir.apply(reward)

        other_rewards = np.array(
            [reward, lin_vel_reward, ang_vel_reward, torque_cost, joint_speed_cost, orientation_cost, smoothness_cost,
             survive_reward, orien_yaw_cost])

        return reward, other_rewards


def K_kernel(x):
    x = np.linalg.norm(x)
    K = -1 / (np.exp(x) + 2 + np.exp(-x))
    return K


def K_kernel2(x):
    x = np.linalg.norm(x)
    x = np.clip(x, -10, 10)
    K = -1 / (np.exp(x / 0.2) + np.exp(-x / 0.2))
    return K

def K_kernel3(x):
    x = np.linalg.norm(x)
    K = -1 / (np.exp(x/0.1) + 2 + np.exp(-x/0.1))
    return K






