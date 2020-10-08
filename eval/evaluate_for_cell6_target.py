import sys
sys.path.append('/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr')


from datetime import datetime
import shutil
import glob
import paramiko
import numpy as np
import pandas as pd
from matplotlib.pylab import plt
from eval.eval_tools import *
from eval.plot_results import *
import os
import argparse
import ast
from baselines.common import plot_util as pu
from eval.instrument import IO

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--env_name', default=None, metavar='G' )
parser.add_argument('--group_dir', type= str, default=None, )
parser.add_argument('--exp_id', type= int, default=None )
parser.add_argument('--full_output', type= ast.literal_eval, default=False )
parser.add_argument('--monitor', type=ast.literal_eval, default=False )

parser.add_argument('--data_name', type= str, default=None, )
parser.add_argument('--global_command', type= str, default=None, )
parser.add_argument('--rand_init', type= int, default=None )
parser.add_argument('--seed', type= int, default=None )
parser.add_argument('--contact_log', type= str, default=None, )
args = parser.parse_args()

root_path = '/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr'
os.chdir(root_path)

seed = 16# 11
global_command = 's2-cell6-xy10-quan' #'cons100'  's1'   s2-cell6-xy10  s2-cell6-10  s2-cell6-xy-circle
rand_init = 0 #
data_name = None#
contact_log = None

# 实验数据原始目录
ENV_name = 'CellRobotEnvCPG6NewTarget-v2'
group_dir = 'log-files-SMC/AWS_logfiles/Oct_07_SMC_PPO_RL_Exp46'
exp_id = 46
exp_no_list= [1]
num_enjoy = 10000
dt = 0.05 # 0.01 for old env(cell4), 0.05 for Cell5 and cell6
max_step = 1000 # 2000 for old env(cell4), 1000 for Cell5 and cell6


model_save_num = None
monitor = args.monitor
render = not monitor
render = True

if args.env_name is None:
    ENV_name = ENV_name
else:
    ENV_name = args.env_name

if args.group_dir is None:
    group_dir = group_dir
else:
    group_dir = args.group_dir

if args.exp_id is None:
    exp_id = exp_id
else:
    exp_id = args.exp_id

if args.data_name is None:
    data_name = data_name
    if data_name is None:
        data_name = '0'
else:
    data_name = args.data_name

if args.contact_log is None:
    contact_log = contact_log

else:
    data_name = args.data_name

if args.global_command is None:
    global_command = global_command
else:
    global_command = args.global_command
if args.rand_init is None:
    rand_init = rand_init
else:
    rand_init = args.rand_init
if args.seed is None:
    seed = seed
else:
    seed = args.seed

exp_path = os.path.join(root_path, group_dir  )
exp_dir_list = os.listdir(exp_path)

exp_list = IO(os.path.join(os.path.join(root_path, group_dir), 'exp_id{}_param.pkl').format(exp_id)).read_pickle()
exp_list_pd = pd.DataFrame(exp_list)

print(exp_list_pd)
exp_list_pd.to_csv(os.path.join(group_dir,'exp_id{}_param.csv'.format(exp_id)), index=True,   header=True)

if args.full_output is False:
    exp_no_list= exp_no_list
else:
    exp_no_list = list(range(1,len(exp_list)))

print(exp_no_list)

results_dir = os.path.join(root_path,  group_dir , 'results')
monitor_dir = os.path.join(root_path,  group_dir ,  'monitor')



if not os.path.isdir(results_dir):
    print('create dir')
    os.makedirs(results_dir)


results_list = list()
last_results = list()

for exp_no in exp_no_list:
    r_path = find_ExpPath(exp_no, exp_dir_list)

    # read results and params
    result_path = os.path.join(exp_path, r_path)
    parms = exp_list_pd['exp{}'.format(exp_no)]

    save_plot_path1 = os.path.join(results_dir, 'No_{}'.format(exp_no))


    # parse results


    reward_res = pu.load_results(result_path)

    # plot learning curve

    plot_learning_curve(reward_res, save_plot_path1, exp_no )

    # evaluate
    evaluate_fun(result_path, parms,model_save_num , num_enjoy=num_enjoy ,global_command=global_command,
                 render = render, monitor = monitor, rand_init= rand_init, seed=seed, data_name = data_name, contact_log = contact_log, env_name=ENV_name)

    eval_path = os.path.join(result_path, 'evaluate')
    eval_data_path = os.path.join(eval_path, 'log_data_{}.csv'.format(data_name))
    eval_data_df = pd.read_csv(eval_data_path)

   # if parms['reward_fun_choice'] == 11:
    v_e = eval_data_df.loc[:, '1':'3'].values
    c_command = eval_data_df.loc[:, '4':'6'].values
    xyz = eval_data_df.loc[:, '7':'9'].values
    # else:
    #     raise Exception('Setting parsing ')


    # save_plot_path2 = os.path.join(eval_path, 'No_{}'.format(exp_no))
    # for save_plot_path in [save_plot_path1, save_plot_path2]:
    #     plot_velocity_curve(v_e, c_command, max_step, dt =dt, save_plot_path=save_plot_path)
    #     plot_position_time(xyz, max_step, dt =dt,save_plot_path=save_plot_path)
    #     plot_traj_xy(xyz, max_step, dt =dt,save_plot_path=save_plot_path)
    #     plot_traj_xy_cmd(xyz,c_command , max_step, dt =dt,save_plot_path=save_plot_path)
    #     plot_cell6_vel_tracking(xyz, v_e, c_command, save_plot_path=save_plot_path)
    #     plot_cell6_vel_tracking_xy(xyz, v_e, c_command, save_plot_path=save_plot_path)

