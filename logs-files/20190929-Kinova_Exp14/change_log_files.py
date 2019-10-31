root_path = '/home/drl/PycharmProjects/JerryRepos/pytorch-a2c-ppo-acktr-gail'
import os
import yaml
import pandas as pd
import shutil

os.chdir(root_path)

exp_NO = 14
log_path = "logs-files/20190929-Kinova_Exp14"

with open(os.path.join(log_path, 'exp_id{}_param.yaml'.format(exp_NO)), 'r') as f:
    exps_params = yaml.load(f)

def find_exp_file_name(exp_no, expdir_list):
    for i in range(len(expdir_list)):
        if 'No' in expdir_list[i]:
            if expdir_list[i].split('_')[1] == str(exp_no):
                return expdir_list[i]
    return FileNotFoundError


reslog_path = os.path.join(log_path, "res_logs")
expdir_list = os.listdir(log_path)


for exp_id in range(1, 93):
    env_reward_name = exps_params['exp{}'.format(exp_id)]['env_name']+'-R{}'.format(exps_params['exp{}'.format(exp_id)]['reward_fun_choice'])

    env_reward_path = os.path.join(reslog_path, env_reward_name)
    if not os.path.exists(env_reward_path):
        os.makedirs(env_reward_path)

    dstName = exps_params['exp{}'.format(exp_id)]['env_name'].split("-")[0]+'R{}'.format(exps_params['exp{}'.format(exp_id)]['reward_fun_choice']) \
              +'-'+ str(exp_id)
    dstResDir = os.path.join(env_reward_path, dstName)

    sourceDir_name = find_exp_file_name(exp_id, expdir_list)

    sourceResDir = os.path.join(log_path, sourceDir_name)

    print('source path :', sourceResDir)
    print('dst path :', dstResDir)
    shutil.copytree(sourceResDir, dstResDir)

