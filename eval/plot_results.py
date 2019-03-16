import numpy as np
import os
from matplotlib.pylab import plt
from baselines.common import plot_util as pu
def plot_velocity_curve(v_e, c_command, max_step = 100, dt =0.01, save_plot_path=None,exp_no=1,figsize=(10, 6)):
    plt.figure(figsize=figsize)
    name = 'No.{} velocity curve'.format(exp_no)

    t = np.arange(0, max_step * dt, dt)
    ax1 = plt.subplot(311)
    plt.plot(t, c_command[:max_step, 0], color='red', label='ref')
    plt.plot(t, v_e[:max_step, 0], '-g', label='real')
    plt.ylim((-0.5,0.8))
    plt.grid()
    plt.title('Vx')

    ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
    plt.plot(t, c_command[:max_step, 1], color='red', label='ref')
    plt.plot(t, v_e[:max_step, 1], '-g', label='real')
    plt.grid()
    plt.title('Vy')
    plt.ylim((-0.5, 0.8))

    ax2 = plt.subplot(313, sharex=ax1, sharey=ax1)
    plt.plot(t, c_command[:max_step, 2], color='red', label='ref')
    plt.plot(t, v_e[:max_step, -1], '-g', label='real')
    plt.grid()
    plt.title('Wyaw')
    plt.ylim((-1.5, 1.5))

    if save_plot_path is not None:
        plt.savefig(save_plot_path + '-vel_t.jpg')
    else:
        plt.show()

def plot_position_time(xyz, max_step = 100, dt =0.01, save_plot_path=None,exp_no=1,figsize=(10, 6)):

    plt.figure(figsize=figsize)
    name = 'No.{} position curve'.format(exp_no)
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
        plt.savefig(save_plot_path + '-pos_t.jpg')
    else:
        plt.show()

def plot_traj_xy(xyz,max_step = 100, dt =0.01,  save_plot_path=None,exp_no=1,figsize=(10, 6)):
    plt.figure(figsize=figsize)
    name = 'No.{} trajectory'.format(exp_no)
    t = np.arange(0, max_step * dt, dt)
    plt.plot(xyz[:max_step, 0], xyz[:max_step, 1], color='red' )
    plt.ylim((-0.5, 0.5))

    plt.grid()
    plt.legend()

    if save_plot_path is not None:
        plt.savefig(save_plot_path + '-xy.jpg')
    else:
        plt.show()


# def plot_learning_curve(rewards_df, save_plot_path=None, exp_no=1, figsize=(8,6)):
#
#     plt.figure(figsize=figsize)
#     name = 'No.{} learning curve'.format(exp_no)
#     plt.plot(rewards_df['total_setps'], rewards_df['AverageCost'])
#
#     plt.title(name)
#    # save_plot_path = None
#     if save_plot_path is not None:
#         plt.savefig(save_plot_path+ '-learning_curve.jpg'  )

def plot_learning_curve(r, save_plot_path=None, exp_no=1, figsize=(8,6)):

    name = 'No.{} learning curve'.format(exp_no)
    #plt.plot(r.progress.TimestepsSoFar, r.progress.EpRewMean)
    fig, ax = pu.plot_results(r, average_group=True, split_fn=lambda _: '', shaded_std=False, figsize=figsize)

    plt.title(name)
    if save_plot_path is not None:
        plt.savefig( save_plot_path+'-learning_curve.jpg' )
    else:
        plt.show()

def plot_traj_xy_cmd(xyz,cmd, max_step = 100, dt =0.01,  save_plot_path=None,exp_no=1,figsize=(10, 6)):
    pos_x = []
    pos_y = []
    x = 0
    y = 0
    for i in range(max_step):
        x += cmd[i, 0] * dt
        y += cmd[i, 1] * dt
        pos_x.append(x)
        pos_y.append(y)

    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)

    plt.figure(figsize=figsize)
    name = 'No.{} trajectory'.format(exp_no)
    t = np.arange(0, max_step * dt, dt)
    plt.plot(xyz[:max_step, 0], xyz[:max_step, 1], 'g' )
    plt.plot(pos_x, pos_y, 'r--')
    #plt.ylim((-0.5, 0.5))

    plt.grid()
    plt.legend()

    if save_plot_path is not None:
        plt.savefig(save_plot_path + '-xy-cmd.jpg')
    else:
        plt.show()