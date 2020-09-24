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
    plt.ylim((-0,0.3))
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


def plot_cell6_vel_tracking(xyz, v_e,c_command,  save_plot_path=None):
    max_step = 2000
    dt = 0.05


    t = np.arange(0, max_step * dt, dt)

    pos_f = []
    for i in range(max_step):
        if i == 0:
            pos = c_command[i, 0]
        else:
            pos += c_command[i, 0] * dt
        pos_f.append(pos)
    pos_f = np.array(pos_f)

    pos = xyz[:max_step, 0]
    vel = v_e[:max_step, 0]
    vel_f = c_command[:max_step, 0]
    pos_error = np.sqrt((pos_f - pos) ** 2).mean()

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, pos)
    axs[0].plot(t, pos_f)
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('X Distance[m]')
    axs[0].grid(True)

    vel_error = np.sqrt((vel_f - vel) ** 2).mean()
    axs[1].plot(t, vel, label='v')
    axs[1].plot(t, vel_f, label='ref')
    axs[1].set_ylim(0, 0.3)
    axs[1].set_xlabel('Time [s], pos err:{:.3f} vel err:{:.3f}'.format(pos_error, vel_error))
    axs[1].set_ylabel('X Velocity [m/s]')
    axs[1].grid(True)

    # fig.tight_layout()

    # plt.savefig(os.path.join(save_fig_path, 'EXP{}-No{}_f2{}.jpg'.format(exp_id, exp_i, exp_dir_list[exp_i])))
    data = [pos_f, pos, vel, vel_f]
    # IO(save_fig_path+'/EXP{}-No{}_f2{}.pkl'.format(exp_id, exp_i, exp_dir_list[exp_i])).to_pickle(data)
    if save_plot_path is not None:
        plt.savefig(save_plot_path + '-cell6-vel.jpg')
    else:
        plt.show()


def plot_cell6_vel_tracking_xy(xyz, v_e,c_command,  save_plot_path=None):
    max_step = 2000
    dt = 0.05

    t = np.arange(0, max_step * dt, dt)

    pos_f = []
    pos_fy = []
    for i in range(max_step):
        if i == 0:
            pos = c_command[i, 0]
            pos_y = c_command[i, 1]
        else:
            pos += c_command[i, 0] * dt
            pos_y += c_command[i, 1] * dt
        pos_f.append(pos)
        pos_fy.append(pos_y)

    pos_f = np.array(pos_f)
    pos_fy = np.array(pos_fy)

    pos = xyz[:max_step, 0]
    pos_y = xyz[:max_step, 1]

    vel = v_e[:max_step, 0]
    vel_f = c_command[:max_step, 0]
    vel_y = v_e[:max_step, 1]
    vel_fy = c_command[:max_step, 1]

    fig, axs = plt.subplots(4, 1)

    pos_error = np.sqrt((pos_f - pos) ** 2).mean()
    axs[0].plot(t, pos)
    axs[0].plot(t, pos_f)
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('X Distance[m]')
    axs[0].grid(True)

    vel_error = np.sqrt((vel_f - vel) ** 2).mean()
    axs[1].plot(t, vel, label='v')
    axs[1].plot(t, vel_f, label='ref')
    axs[1].set_ylim(0, 0.3)
    #axs[1].set_xlabel('Time [s], pos err:{:.3f} vel err:{:.3f}'.format(pos_error, vel_error))
    axs[1].set_ylabel('X Velocity [m/s]')
    axs[1].grid(True)

    pos_y_error = np.sqrt((pos_fy - pos_y) ** 2).mean()
    axs[2].plot(t, pos_y)
    axs[2].plot(t, pos_fy)
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('X Distance[m]')
    axs[2].grid(True)

    vel_y_error = np.sqrt((vel_fy - vel_y) ** 2).mean()
    axs[3].plot(t, vel_y, label='v')
    axs[3].plot(t, vel_fy, label='ref')
    axs[3].set_ylim(0, 0.3)
    axs[3].set_xlabel('Time [s], pos[x,y] err:{:.3f},{:.3f} vel[x,y] err:{:.3f} {:.3f}'.format(pos_error,pos_y_error, vel_error, vel_y_error))
    axs[3].set_ylabel('X Velocity [m/s]')
    axs[3].grid(True)

    # fig.tight_layout()


    # IO(save_fig_path+'/EXP{}-No{}_f2{}.pkl'.format(exp_id, exp_i, exp_dir_list[exp_i])).to_pickle(data)
    if save_plot_path is not None:
        plt.savefig(save_plot_path + '-cell6-vel-xy.jpg')
    else:
        plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))

    axs.plot(pos_f, pos_fy, label='v')
    axs.plot(pos, pos_y, label='ref')

    pos_ref = np.concatenate((pos_f[None], pos_fy[None]), axis= 0).transpose()
    pos_true = np.concatenate((pos[None], pos_y[None]), axis=0).transpose()

    tracking_error = np.linalg.norm(pos_ref - pos_true, axis= 1).mean()
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.grid(True)
    axs.set_title('error : {:.3f}'.format(tracking_error))

    if save_plot_path is not None:
        plt.savefig(save_plot_path + '-cell6-xy-pos.jpg')
    else:
        plt.show()