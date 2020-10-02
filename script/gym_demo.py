
import gym
import os
import numpy as np
import cv2
#----
from my_envs.mujoco import *
#---

#os.environ["XML_NAME"] = "cellrobot_Quadruped_float_limit_ball.xml"

os.environ["CPG_ENABLE"] = str(1)

os.environ["NUM_BUFFER"] = str(0)
os.environ["COMMAND_MODE"] = "FandE"
os.environ["VEL_FILTER"] = str(1)
os.environ["REWARD_CHOICE"] = str(0)

os.environ["GLOBAL_CMD"] = 's2-cell6'

os.environ["COMMAND_X"] = str(0.2)
os.environ["COMMAND_Y"] = str(0.2)
os.environ["COMMAND_Z"] = str(0)
os.environ["ACTION_DIM"] = str(2)

# os.environ["SAMPLE_MODE"] = "1"
# os.environ["COMMAND_MODE"] = "point"  #point dir_vel
os.environ["COMMAND_MODE"] = "conv_error"  #point dir_vel
#HalfCheetah-v2  CellrobotEnvCPG5-v0 Ant-v2  CellRobotEnvCPG6Goal-v1 CellRobotEnvCPG6Traj-v1 CellRobotEnvCPG6Traj-v1  CellRobotEnvCPG6Target-v2
env = gym.make("CellRobotEnvCPG6Traj-v3" )


obs = env.reset()

print("obs:", env.observation_space )
print("action:", env.action_space.shape)

t=0
for i in range(5000):
    print('t={}'.format(t))
    n_dim_action = env.action_space.shape[0]
    action = env.action_space.sample() # np.zeros(n_dim_action)#
    obs, reward, done, info = env.step(action)

    t += 1
    if done :
        print("reset:")
        env.reset()
        t = 0


    env.render()




