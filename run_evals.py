import sys
import os
import argparse
import ast
from baselines.common import plot_util as pu
import yaml
from evals.utils import find_ExpPath
from evals.evalfuns import plot_learning_curve, evaluate_fun

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--group_dir', type= str, default=None, )
parser.add_argument('--exp_id', type= int, default=None )

#
args = parser.parse_args()
root_path = os.path.dirname(os.path.abspath(__file__))

# 实验数据原始目录
group_dir = 'logs-files/20190919-Kinova_Exp3'
exp_id = 3
seed = 1
exp_no_list= [ 2,   ]
isRender = True
num_enjoy = 1

with open(os.path.join(group_dir, 'exp_id{}_param.yaml'.format(exp_id))) as f:
    exps_params = yaml.load(f)

exp_path = os.path.join(root_path, group_dir  )
exp_dir_list = os.listdir(exp_path)


results_dir = os.path.join(root_path,  group_dir , 'results')
monitor_dir = os.path.join(root_path,  group_dir ,  'monitor')

if not os.path.isdir(results_dir):
    print('creating some folders.')
    os.makedirs(results_dir)


results_list = list()
last_results = list()

for exp_no in exp_no_list:
    r_path = find_ExpPath(exp_no, exp_dir_list)

    # read results and params
    result_path = os.path.join(exp_path, r_path)
    parms = exps_params['exp{}'.format(exp_no)]

    save_plot_path1 = os.path.join(results_dir, 'No_{}'.format(exp_no))

    # parse results

    reward_res = pu.load_results(result_path)

    # plot learning curve
    plot_learning_curve(reward_res, save_plot_path1, exp_no )

    config = {
        'seed': seed,
        'log_interval':1,
        'isRender':isRender,
        'model_save_num':None, # the name id of model path
    }
    # evaluate
    evaluate_fun(result_path, parms, config)

    eval_path = os.path.join(result_path, 'evaluate')


