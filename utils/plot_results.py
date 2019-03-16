import numpy as np

from matplotlib.pylab import plt

def plot_velocity_curve(v_e, c_command, max_step = 100, dt =0.01):


    t = np.arange(0, max_step * dt, dt)
    ax1 = plt.subplot(311)
    plt.plot(t, c_command[:max_step, 0], color='red', label='ref')
    plt.plot(t, v_e[:max_step, 0], '-g', label='real')
    plt.grid()
    plt.title('Vx')

    ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
    plt.plot(t, c_command[:max_step, 1], color='red', label='ref')
    plt.plot(t, v_e[:max_step, 1], '-g', label='real')
    plt.grid()
    plt.title('Vy')

    ax2 = plt.subplot(313, sharex=ax1, sharey=ax1)
    plt.plot(t, c_command[:max_step, 2], color='red', label='ref')
    plt.plot(t, v_e[:max_step, 2], '-g', label='real')
    plt.grid()
    plt.title('Wyaw')

    plt.show()

def plot_position_time(xyz, max_step = 100, dt =0.01, save_plot_path=None):

    plt.Figure(figsize=(15, 10))

    t = np.arange(0, max_step * dt, dt)
    ax1 = plt.subplot(311)
    plt.plot(t, xyz[:max_step, 0], color='red', label='x')

    plt.grid()
    plt.legend()

    ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
    plt.plot(t, xyz[:max_step, 1], color='red', label='y')
    plt.grid()
    plt.legend()

    ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
    plt.plot(t, xyz[:max_step, 2], color='red', label='z')
    plt.grid()
    plt.legend()
    if save_plot_path is not None:
        plt.savefig(save_plot_path + '_pos_t.jpg')
    else:
        plt.show()

def plot_traj_xy(xyz,max_step = 100, dt =0.01,  save_plot_path=None):
    plt.figure(figsize=(15, 10))

    t = np.arange(0, max_step * dt, dt)
    plt.plot(xyz[:max_step, 0], xyz[:max_step, 1], color='red' )
    plt.ylim((-0.5, 0.5))

    plt.grid()
    plt.legend()

    if save_plot_path is not None:
        plt.savefig(save_plot_path + '_xy.jpg')
    else:
        plt.show()