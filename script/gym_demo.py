
import gym
import os
import numpy as np
import cv2
#----
from my_envs.mujoco import *
import time


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
os.environ["COMMAND_MODE"] = "point"  #point dir_vel
#HalfCheetah-v2  CellrobotEnvCPG5-v0 Ant-v2  CellRobotEnvCPG6Goal-v1 CellRobotEnvCPG6Traj-v1 CellRobotEnvCPG6Traj-v1  CellRobotEnvCPG6Target-v2
env = gym.make("CellRobotEnvCPG6Target-v2" )


obs = env.reset()

print("obs:", env.observation_space )
print("action:", env.action_space.shape)

step_times = []

t=0
for i in range(5000):
    print('t={}'.format(t))
    n_dim_action = env.action_space.shape[0]
    action = env.action_space.sample() # np.zeros(n_dim_action)#

    t_start = time.time()
    obs, reward, done, info = env.step(action)
    t_end = time.time()

    step_times.append(t_end - t_start)
    #print("step time: {}", t_end - t_start)

    t += 1
    if done :
        print("reset:")

        print("average step time: {:.5s} s", sum(step_times)/len(step_times))
        step_times = []

        env.reset()
        t = 0


    env.render()




