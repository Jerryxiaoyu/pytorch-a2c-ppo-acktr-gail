root_path = '/home/drl/PycharmProjects/JerryRepos/pytorch-a2c-ppo-acktr-gail'
import os

os.chdir(root_path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baselines.common import plot_util as pu




save_fig_path = 'logs-files/20190929-Kinova_Exp14/results'

RL_file_name = "KinovaCupPusherEnv-v0-R1"

for RL_file_name in os.listdir("logs-files/20190929-Kinova_Exp14/res_logs"):
    RL_data_path = "logs-files/20190929-Kinova_Exp14/res_logs/" +RL_file_name
    fig_name = RL_file_name

    result_path = os.path.join(RL_data_path)
    r2 = pu.load_results(result_path)

    figsize = (8, 6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    try:
        fig, ax = pu.my_plot_results(r2, average_group=True, split_fn=lambda _: '', shaded_std=False,
                                            legend_outside=True, figsize=figsize, logx_fig=True )
        ax = ax[0, 0]

        # 字体设置
        font1 = {'family': 'DejaVu Sans',
                 'weight': 'normal',
                 'size': 8,
                 }

        ax.set_xlabel('Steps', font1)
        ax.set_ylabel('Reward ', font1)

        ax.set_xticks([1e4, 1e5, 1e6, 1e7])

        plt.tick_params(labelsize=8)
        # plt.grid()

        fig.set_size_inches(8.2 / 2.54, 8.2 / 2.54 / (1.618))
        fig.subplots_adjust(left=0.18, bottom=0.18)
        plt.savefig(save_fig_path + '/{}-learning_curve_logx.svg'.format(fig_name), dpi=600, pad_inches=0)
    except Exception:
        pass
    #plt.show()