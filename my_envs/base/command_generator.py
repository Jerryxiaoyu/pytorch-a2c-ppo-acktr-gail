import numpy as np
from matplotlib.pylab import plt



def command_generator(max_step, dt,  hold_time, delta=0.025, vx_range = (-0.8, 0.8), vy_range = (-0.8, 0.8), wyaw_range = (-0.8, 0.8),render = False, seed = None):
    if seed is not None:
        np.random.seed(seed)
    vx_range = vx_range
    vy_range = vy_range
    wyaw_range = wyaw_range

    command_vx = []
    command_vy = []
    command_wyaw = []

    num_points = int(max_step/int(hold_time/dt))

    vx_p_range = int(vx_range[1]*10)
    vy_p_range = int(vy_range[1] * 10)
    wyaw_p_range = int(wyaw_range[1] * 10)

    vx_p_range = int((vx_range[1] - vx_range[0])/delta)
    vy_p_range = int((vy_range[1] - vy_range[0]) / delta)
    wyaw_p_range = int((wyaw_range[1] - wyaw_range[0]) / delta)

    # vx_p = np.random.uniform(vx_range[0], vx_range[1], num_points)
    # vy_p = np.random.uniform(vy_range[0], vy_range[1], num_points)
    # wyaw_p = np.random.uniform(wyaw_range[0], wyaw_range[1], num_points)

    vx_p = np.random.randint(0, vx_p_range + 1, num_points) * delta + vx_range[0]
    vy_p = np.random.randint(0, vy_p_range + 1, num_points) * delta + vy_range[0]
    wyaw_p =np.random.randint(0, wyaw_p_range + 1, num_points) * delta + wyaw_range[0]

    step = 0
    p_index =0
    time_done =0
    while step < max_step:

        if time_done < hold_time:
            command_vx.append(vx_p[p_index])
            command_vy.append(vy_p[p_index])
            command_wyaw.append(wyaw_p[p_index])
        else:
            time_done =0
            p_index += 1

        time_done += dt
        step +=1



    command_vx = np.array(command_vx)
    command_vy = np.array(command_vy)
    command_wyaw = np.array(command_wyaw)



    command = np.array([command_vx, command_vy, command_wyaw]).T
    if render:
        ax1 = plt.subplot(311)
        plt.plot(command_vx, color='red', label='o_1')
        plt.grid()
        plt.title('Vx')


        ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
        plt.plot(command_vy, color='red', label='o_1')
        plt.grid()
        plt.title('Vy')

        ax2 = plt.subplot(313, sharex=ax1, sharey=ax1)
        plt.plot(command_wyaw, color='red', label='o_1')
        plt.grid()
        plt.title('Wyaw')


        plt.show()


    return  command


def command_evaluate_generator(max_step,dt,  hold_time, render = False):


    vx_range = (-0.5, 0.5)
    vy_range = (0, 0)
    wyaw_range = (0, 0)

    command_vx = []
    command_vy = []
    command_wyaw = []

    num_points = int(max_step/int(hold_time/dt))

    vx_p = np.random.uniform(vx_range[0], vx_range[1], num_points)
    vy_p = np.random.uniform(vy_range[0], vy_range[1], num_points)
    wyaw_p = np.random.uniform(wyaw_range[0], wyaw_range[1], num_points)

    step = 0
    p_index =0
    time_done =0
    while step < max_step:

        if time_done < hold_time:
            command_vx.append(vx_p[p_index])
            command_vy.append(vy_p[p_index])
            command_wyaw.append(wyaw_p[p_index])
        else:
            time_done =0
            p_index += 1

        time_done += dt
        step +=1



    command_vx = np.array(command_vx)
    command_vy = np.array(command_vy)
    command_wyaw = np.array(command_wyaw)



    command = np.array([command_vx, command_vy, command_wyaw]).T

    if render:
        ax1 = plt.subplot(311)
        plt.plot(command_vx, color='red', label='o_1')
        plt.grid()
        plt.title('Vx')


        ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
        plt.plot(command_vy, color='red', label='o_1')
        plt.grid()
        plt.title('Vy')

        ax2 = plt.subplot(313, sharex=ax1, sharey=ax1)
        plt.plot(command_wyaw, color='red', label='o_1')
        plt.grid()
        plt.title('Wyaw')


        plt.show()


    return  command

def plot_command(command):
    command_vx = command[:,0]
    command_vy = command[:,1]
    command_wyaw = command[:,2]
    ax1 = plt.subplot(311)
    plt.plot(command_vx, color='red', label='o_1')
    plt.grid()
    plt.title('Vx')


    ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
    plt.plot(command_vy, color='red', label='o_1')
    plt.grid()
    plt.title('Vy')

    ax2 = plt.subplot(313, sharex=ax1, sharey=ax1)
    plt.plot(command_wyaw, color='red', label='o_1')
    plt.grid()
    plt.title('Wyaw')


    plt.show()

#
# for n, g in enumerate(command_generator(1000, 0.01,  2)):
#     plt.scatter(n,g)
#
# plt.show()


#command_generator(2000, 0.01, 4, vx_range=(0, 0.4), vy_range=(0, 0.4), wyaw_range=(0, 0), render=True)

#command_evaluate_generator(10000, 0.01,  4)