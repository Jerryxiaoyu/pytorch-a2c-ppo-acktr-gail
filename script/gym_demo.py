
import gym
import os
import numpy as np
import cv2
#----
from my_envs.mujoco import *
#---

os.environ["CPG_ENABLE"] = str(1)

os.environ["NUM_BUFFER"] = str(0)
os.environ["COMMAND_MODE"] = "FandE"
os.environ["VEL_FILTER"] = str(0)
os.environ["REWARD_CHOICE"] = str(6)

os.environ["GLOBAL_CMD"] = 's2-cell6'

os.environ["COMMAND_X"] = str(0.2)
os.environ["COMMAND_Y"] = str(0.2)
os.environ["COMMAND_Z"] = str(0)



os.environ["ACTION_DIM"] = str(2)

#HalfCheetah-v2  CellrobotEnvCPG5-v0 Ant-v2  CellRobotEnvCPG6Goal-v1 CellRobotEnvCPG6Traj-v1
env = gym.make("CellRobotEnvCPG6Traj-v2"  )


obs = env.reset()

print("obs:", env.observation_space )
print("action:", env.action_space.shape)


for i in range(5000):
    n_dim_action = env.action_space.shape[0]
    action = env.action_space.sample() # np.zeros(n_dim_action)#
    obs, reward, done, info = env.step(action)

    if done :
        print("reset:")
        env.reset()

    env.render()




