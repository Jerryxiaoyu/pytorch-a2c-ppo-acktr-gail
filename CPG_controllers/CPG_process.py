
import numpy as np
import matplotlib


import matplotlib.pyplot as plt
import time
from utils import log

import os
import gym
from CPG_controllers.fitness import calc_fitness
import datetime


def position_PID(cur_angles, target_angles, target_velocities=0):
    #target_angles = target_angles.reshape((-1, 1))

    #q = cur_angles.reshape((-1, 1))

    kp = 5

    Kp = np.diag(np.ones(cur_angles.shape[0]) * kp)

    action = Kp.dot(target_angles - cur_angles)

    action = np.clip(action, -1, 1)
    action = np.array(action)

    return action.reshape((1, -1))[0]


def oscillator_nw( position_vector,robot, dt=0.01, max_time=10.0, fitness_option=6, plot=False, log_dis=False,
                  render=False, monitor_path=None, save_plot_path=None ):
    if log_dis:
        log.infov('[OSC]-------------------------------------------------------------')
        log.infov('[OSC] Run in multiprocessing({})'.format(os.getpid()))

        log.infov('[OSC] Running oscillator_2.oscillator_nw')
        log.info('[OSC] Printing chromosome')
        log.info('[OSC] {0}'.format(position_vector))
        log.info('[OSC] Started monitoring thread')

    env = robot.env
    obs_low = robot.obs_pos_index_l
    obs_high = robot.obs_pos_index_h
    height_threshold = robot.height_threshold


    def get_position(obs):
        return obs[obs_low:obs_high]

    # For plots - not needed now
    if plot:
        o1_list = list()
        o2_list = list()
        o3_list = list()
        o4_list = list()
        o5_list = list()
        o6_list = list()
        o7_list = list()
        o8_list = list()
        o9_list = list()
        o10_list = list()
        o11_list = list()
        o12_list = list()
        o13_list = list()

        end_pos_x_list = list()
        end_pos_y_list = list()
        end_pos_z_list = list()
        t_list = list()

    CPG_controller = robot.CPG_controller(CPG_node_num=robot.num_cell, position_vector=position_vector, dt=dt)


    max_step = int(max_time / dt)

    if monitor_path is not None:
        env = gym.wrappers.Monitor(env, monitor_path, force=True)

    obs = env.reset()

    start = datetime.datetime.now()

    initial_bias_angles = position_vector[1+robot.num_cell:1+robot.num_cell*2]

    t0 = time.time()
    for i in range(int(2 / dt)):
        joint_postion_ref = np.array(initial_bias_angles)
        cur_angles = get_position(obs)
        action = position_PID(cur_angles, joint_postion_ref, target_velocities=0)

        # execute action
        next_state, reward, done, _ = env.step(action)

        obs = next_state

        if done is True:
            break
        #env.render()

    t1 = time.time()
    #print('daole! = ', t1 - t0)
    start_pos_x = obs[0]
    start_pos_y = obs[1]
    start_pos_z = obs[2]

    obs_list = list()
    base_z_list = list()
    end_pos_xyz = list()

    up_time = 0
    fallen = False
    t0 = time.time()
    for i in range(max_step):
        output_list = CPG_controller.output(state=None)

        joint_postion_ref = np.array(output_list[1:])
        cur_angles = get_position(obs)
        action = position_PID(cur_angles, joint_postion_ref, target_velocities=0)

        # execute action
        action += np.random.normal(0, 0.03, action.shape )
        next_state, reward, done, _ = env.step(action)

        obs = next_state
        obs_list.append(obs)


        avg_z = obs[2]
        base_z = obs[2]
        base_z_list.append(base_z)
        if base_z < height_threshold or done is True:
            # Set the flag which indicates that the robot has fallen
            fallen = True
            # Calculate the average height
            avg_z = sum(base_z_list) / float(len(base_z_list))

            break

        if render:
            env.render()

        if monitor_path is not None  :
            env.render(mode='rgb_array')

        up_time += dt
        end_pos_xyz.append(obs[:3])
        # For plots - not needed now
        if plot:
            o1_list.append(output_list[1])
            o2_list.append(output_list[2])
            o3_list.append(output_list[3])
            o4_list.append(output_list[4])
            o5_list.append(output_list[5])
            o6_list.append(output_list[6])
            o7_list.append(output_list[7])
            o8_list.append(output_list[8])
            o9_list.append(output_list[9])
            o10_list.append(output_list[10])
            o11_list.append(output_list[11])
            o12_list.append(output_list[12])
            o13_list.append(output_list[13])

            end_pos_x_list.append(obs[0])
            end_pos_y_list.append(obs[1])
            end_pos_z_list.append(obs[2])
            t_list.append(i * dt)




    t1 = time.time()
    #print('During time is ', t1 - t0)
    # Outside the loop, it means that either the robot has fallen or the max_time has elapsed
    # Find out the end position of the robot
    end_pos_x = obs[0]
    end_pos_y = obs[1]
    end_pos_z = obs[2]

    if monitor_path is not None:
        env.close()


    # Find the up time
    up_time = min(up_time, max_step * dt)

    # Calculate the fitness
    if up_time == 0.0:
        fitness = 0.0
        if log_dis:
            log('[OSC] up_t==0 so fitness is set to 0.0')
    else:
        fitness = calc_fitness(start_x=start_pos_x, start_y=start_pos_y, start_z=start_pos_z,
                               end_x=end_pos_x, end_y=end_pos_y, end_z=end_pos_z,
                               avg_z=avg_z,
                               up_time=up_time,
                               end_pos_xyz = np.array(end_pos_xyz),
                               fitness_option=fitness_option
                               )

    if log_dis:
        if not fallen:
            log.info("[OSC] Robot has not fallen")
        else:
            log.info("[OSC] Robot has fallen")

        log.info('[OSC] Calculated fitness: {0}'.format(fitness))

    x_distance = end_pos_x - start_pos_x

    abs_y_deviation = end_pos_y
    avg_footstep_x = None
    var_torso_alpha = obs[3]
    var_torso_beta = obs[4]
    var_torso_gamma = obs[5]

    # For plots - not needed now
    if plot:
        ax1 = plt.subplot(611)
        plt.plot(t_list, o1_list, color='red', label='o_1')
        plt.plot(t_list, o2_list, color='green', ls='--', label='o_2')
        plt.plot(t_list, o3_list, color='green', label='o_3')
        plt.grid()
        plt.legend()

        ax2 = plt.subplot(612, sharex=ax1, sharey=ax1)
        plt.plot(t_list, o1_list, color='red', label='o_1')
        plt.plot(t_list, o4_list, color='blue', ls='--', label='o_4')
        plt.plot(t_list, o5_list, color='blue', label='o_5')
        plt.grid()
        plt.legend()

        ax3 = plt.subplot(613, sharex=ax1, sharey=ax1)
        plt.plot(t_list, o1_list, color='red', label='o_1')
        plt.plot(t_list, o6_list, color='black', ls='--', label='o_6')
        plt.plot(t_list, o7_list, color='black', label='o_7')
        plt.grid()
        plt.legend()

        ax4 = plt.subplot(614, sharex=ax1, sharey=ax1)
        plt.plot(t_list, o1_list, color='red', label='o_1')
        plt.plot(t_list, o8_list, color='cyan', ls='--', label='o_8')
        plt.plot(t_list, o9_list, color='cyan', label='o_9')
        plt.grid()
        plt.legend()

        ax5 = plt.subplot(615, sharex=ax1, sharey=ax1)
        plt.plot(t_list, o1_list, color='red', label='o_1')
        plt.plot(t_list, o10_list, color='orange', ls='--', label='o_10')
        plt.plot(t_list, o11_list, color='orange', label='o_11')
        plt.grid()
        plt.legend()

        ax6 = plt.subplot(616, sharex=ax1, sharey=ax1)
        plt.plot(t_list, o1_list, color='red', label='o_1')
        plt.plot(t_list, o12_list, color='brown', ls='--', label='o_12')
        plt.plot(t_list, o13_list, color='brown', label='o_13')
        plt.grid()
        plt.legend()
        if save_plot_path is not None:
            plt.savefig(save_plot_path + '_cpg.jpg')
        else:
            plt.show()

        plt.Figure(figsize=(15, 10))

        ax1 = plt.subplot(311)
        plt.plot(t_list, end_pos_x_list, color='red', label='x')

        plt.grid()
        plt.legend()

        ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
        plt.plot(t_list, end_pos_y_list, color='red', label='y')
        plt.grid()
        plt.legend()

        ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
        plt.plot(t_list, end_pos_z_list, color='red', label='z')
        plt.grid()
        plt.legend()
        if save_plot_path is not None:
            plt.savefig(save_plot_path + '_pos.jpg')
        else:
            plt.show()



        plt.figure(figsize=(15, 10))

        plt.plot(end_pos_x_list, end_pos_y_list, color='red' )
        plt.ylim((-0.5, 0.5))

        plt.grid()
        plt.legend()

        if save_plot_path is not None:
            plt.savefig(save_plot_path + '_traj.jpg')
        else:
            plt.show()


    # Different from original script
    # Return the evaluation metrics
    return {'fitness': fitness,
            'fallen': fallen,
            'up': up_time,
            'x_distance': x_distance,
            'abs_y_deviation': abs_y_deviation,
            'avg_footstep_x': avg_footstep_x,
            'var_torso_alpha': var_torso_alpha,
            'var_torso_beta': var_torso_beta,
            'var_torso_gamma': var_torso_gamma}




def test_CPGprocess():
    from robot_discription.CRbot import CRbot
    env = gym.make('CellrobotEnv-v0')
    # sinusoid  40  matsuoka 27    sinusoid_mix 53   matsuoka_mix 40
    robot = CRbot(env, CPG_type = 'sinusoid_mix')

    position_vector = np.zeros(53)
    position_vector[0] = 1
    for i in range(1, 14):
        position_vector[i] = 1

    result = oscillator_nw(position_vector,robot, dt=0.01, plot=False, render=True, monitor_path=None,
                           # '/home/drl/PycharmProjects/DeployedProjects/CR_CPG/tmp/tmp2.mp4'
                           save_plot_path=None)  # '/home/drl/PycharmProjects/DeployedProjects/CR_CPG/tmp/tmp.mp4'

    print('result = ', result)


