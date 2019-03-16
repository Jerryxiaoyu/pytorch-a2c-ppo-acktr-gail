import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env
import os

reward_choice = os.getenv('REWARD_CHOICE')


class MyAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self.Calc_Reward = self.reward_fun1

        #mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 1)
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 1)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)

        xposafter = self.get_body_com("torso")[0]
        forward_vel = (xposafter - xposbefore) / self.dt

        v_commdand = 0
        reward, other_rewards = self.Calc_Reward(forward_vel, v_commdand, a, None)

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            velocity_base=forward_vel,
            commands=v_commdand,
            rewards=other_rewards,
            obs=ob)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self,command = None, reward_fun_choice = None):

        global reward_choice
        if reward_choice is None:
            print('REWARD_CHOICE is not specified!')
            reward_fun_choice_env = 1
        else:
            reward_fun_choice_env = int(reward_choice)

        if reward_fun_choice is None:
            reward_fun_choice = reward_fun_choice_env

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

        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reward_fun1(self, forward_vel, v_commdand, action, obs):

        forward_reward = forward_vel
        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        other_rewards = np.array([reward, forward_reward, -ctrl_cost, -contact_cost, survive_reward])

        return reward, other_rewards

    def reward_fun2(self, forward_vel, v_commdand, action, obs):
        # print('reward2')
        forward_reward = forward_vel
        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        other_rewards = np.array([reward, forward_reward, -ctrl_cost, -contact_cost, survive_reward])

        return reward, other_rewards
