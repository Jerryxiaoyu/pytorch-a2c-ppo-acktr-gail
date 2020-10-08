from utils.instrument import VariantGenerator, variant, IO
import os
from datetime import datetime
import shutil
import glob
import paramiko
import time
import utils.ssh as ssh
import argparse
import math

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--exp_id', type= int, default=18  )
parser.add_argument('--n_cpu', type= int, default=8 )

args = parser.parse_args()


class VG(VariantGenerator):

    @variant
    def env_name(self):
        return ['CellRobotEnvCPG6Traj-v2']  # 'CellrobotEnv-v0' , 'Cellrobot2Env-v0', 'CellrobotSnakeEnv-v0'  , 'CellrobotSnake2Env-v0','CellrobotButterflyEnv-v0', 'CellrobotBigdog2Env-v0'

    @variant
    def seed(self):
        return [123]

    @variant
    def num_steps(self):
        return [2048]

    @variant
    def learning_rate(self):
        return [3e-4]  #1e-3

    @variant
    def entropy_coef(self):
        return [0]

    @variant
    def value_loss_coef(self):
        return [0.5]

    @variant
    def ppo_epoch(self):
        return [10]
    @variant
    def num_mini_batch(self):
        return [64]

    @variant
    def gamma(self):
        return [0.99 ] #0.9985

    @variant
    def tau(self):
        return [0.95 ]  #0.98

    @variant
    def num_env_steps(self):
        return [5e7]

    @variant
    def recurrent(self):
        return [0]
 ##----------------------------------------------------
    @variant
    def state_mode(self):
        return ['meiyong', ]  # vel , pos, vel_f

    @variant
    def command_vx_high(self):
        return [0.2]

    @variant
    def command_vy_high(self):
        return [0.2]

    @variant
    def command_vx_low(self):
        return [0]

    @variant
    def command_vy_low(self):
        return [0]

    @variant
    def command_wz_high(self):
        return [0]  # vel , pos

    @variant
    def turing_flag(self):
        return [0]  # 2 line tracking for 5s, 3 for 20s, 1 turning tracking

    @variant
    def xml_name(self):
        return ['cellrobot_Quadruped_float_limit.xml']

    @variant
    def buffer_mode(self):
        return ['meiyong']

    # ----------------------------------------------
    @variant
    def base(self):
        return ['MLPBase256'  ]  # CellRobotMLPBase MLPBase

    @variant
    def action_dim(self):
        return [2 ]  # 2,3,13

    @variant
    def CPG_enable(self):
        return [1 ]

    @variant
    def num_buffer(self):
        return [0 ]

    @variant
    def command_mode(self):
        return [ "FandE" ]  # full, error, no FandE  point  conv_error

    @variant
    def reward_fun_choice(self):
        return [3]

    @variant
    def vel_filtered(self):
        return [1]

    @variant
    def trained_model_path(self):
        return [None]

    @variant
    def sample_mode(self):
        return [None]


exp_id = args.exp_id

EXP_NAME ='_SMC_PPO_RL'
group_note ="************ABOUT THIS EXPERIMENT****************\n" \
            "  " \
        " "
no_cuda = True
sync_s3 = True
inner_upload_s3 = True

n_cpu = args.n_cpu
num_threads = n_cpu

bucket_path = "jaco-bair/cellrobot/AWS_logfiles"

# print choices
variants = VG().variants()
num=0
for v in variants:
    num +=1
    print('exp{}: '.format(num), v)

# save gourp parms
exp_group_dir = datetime.now().strftime("%b_%d")+EXP_NAME+'_Exp{}'.format(exp_id)
group_dir = os.path.join('log-files-SMC', exp_group_dir)
os.makedirs(group_dir)

variants = VG().variants()
num = 0
param_dict = {}
for v in variants:
    num += 1
    print('exp{}: '.format(num), v)
    parm = v
    parm = dict(parm, **v)
    param_d = {'exp{}'.format(num): parm}
    param_dict.update(param_d)


IO('log-files-SMC/' + exp_group_dir + '/exp_id{}_param.pkl'.format(exp_id)).to_pickle(param_dict)
print(' Parameters is saved : exp_id{}_param.pkl'.format(exp_id))
# save args prameters
with open(group_dir + '/readme.txt', 'wt') as f:
    print("Welcome to Jerry's lab\n", file=f)
    print(group_note, file=f)


algo ='ppo'

log_interval = 1
save_model_interval = 50

full_output = True
evaluate_monitor = False


# run
num_exp =0
for v in variants:
    num_exp += 1
    print(v)
    time_now = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    # load parm
    exp_name = 'No_{}_{}_PPO-{}'.format(num_exp, v['env_name'], time_now)
    log_dir =  os.path.join(group_dir,exp_name)
    save_dir =  os.path.join(log_dir,'model')
    seed = v['seed']
    env_name = v['env_name']


    gamma = v['gamma']
    tau = v['tau']

    num_steps = v['num_steps']
    learning_rate = v['learning_rate']
    entropy_coef = v['entropy_coef']
    value_loss_coef = v['value_loss_coef']
    ppo_epoch = v['ppo_epoch']
    num_env_steps = int(v['num_env_steps'])
    num_mini_batch = v['num_mini_batch']

    base = v['base']



    os.environ["REWARD_CHOICE"] = str(v['reward_fun_choice'])
    print('REWARD_CHOICE = ', os.getenv('REWARD_CHOICE'))

    if v['action_dim'] is not None:
        os.environ["ACTION_DIM"] = str(v['action_dim'])
        print('ACTION_DIM = ', os.getenv('ACTION_DIM'))

    if v['num_buffer'] is not None:
        os.environ["NUM_BUFFER"] = str(v['num_buffer'])
        print('NUM_BUFFER = ', os.getenv('NUM_BUFFER'))

    if v['command_mode'] is not None:
        os.environ["COMMAND_MODE"] = str(v['command_mode'])
        print('COMMAND_MODE = ', os.getenv('COMMAND_MODE'))

    if v['buffer_mode'] is not None:
        os.environ["BUFFER_MODE"] = str(v['buffer_mode'])
        print('BUFFER_MODE = ', os.getenv('BUFFER_MODE'))

    if v['CPG_enable'] is not None:
        os.environ["CPG_ENABLE"] = str(v['CPG_enable'])
        print('CPG_ENABLE = ', os.getenv('CPG_ENABLE'))

    if v['state_mode'] is not None:
        os.environ["STATE_MODE"] = str(v['state_mode'])
        print('STATE_MODE = ', os.getenv('STATE_MODE'))


    if v['command_vx_high'] is not None:
        os.environ["COMMAND_X"] = str(v['command_vx_high'])
        print('COMMAND_X = ', os.getenv('COMMAND_X'))

    if v['command_vy_high'] is not None:
        os.environ["COMMAND_Y"] = str(v['command_vy_high'])
        print('COMMAND_Y = ', os.getenv('COMMAND_Y'))

    if v['command_vx_low'] is not None:
        os.environ["COMMAND_X_LOW"] = str(v['command_vx_low'])
        print('COMMAND_X_LOW = ', os.getenv('COMMAND_X_LOW'))

    if v['command_vy_low'] is not None:
        os.environ["COMMAND_Y_LOW"] = str(v['command_vy_low'])
        print('COMMAND_Y_LOW = ', os.getenv('COMMAND_Y_LOW'))


    if v['command_wz_high'] is not None:
        os.environ["COMMAND_Z"] = str(v['command_wz_high'])
        print('COMMAND_Z = ', os.getenv('COMMAND_Z'))

    if v['vel_filtered'] is not None:
            os.environ["VEL_FILTER"] = str(v['vel_filtered'])
            print('VEL_FILTER = ', os.getenv('VEL_FILTER'))

    if v['turing_flag'] is not None:
            os.environ["TURING_FLAG"] = str(v['turing_flag'])
            print('TURING_FLAG = ', os.getenv('TURING_FLAG'))
    if v['xml_name'] is not None:
            os.environ["XML_NAME"] = str(v['xml_name'])
            print('XML_NAME = ', os.getenv('XML_NAME'))

    if v['sample_mode'] is not None:
        os.environ["SAMPLE_MODE"] = str(v['sample_mode'])
        print('SAMPLE_MODE = ', os.getenv('SAMPLE_MODE'))


    if v['recurrent'] == 1 :
        other_str = " --recurrent-policy "
    else:
        other_str = " "

    trained_model_path = v['trained_model_path']
    if trained_model_path is not None:
        other_str += (" --tained-mode-path " + str(trained_model_path) + " ")

    if  inner_upload_s3:
        other_str +=   (" --s3-path "+str(bucket_path)+ " ")
        other_str += (" --upload-s3 ")

    if no_cuda:
        other_str += "  --no-cuda "


    os.system("python3 main.py "  +
              " --env-name " + str(env_name) +
              " --algo " + str(algo) +
              " --use-gae "  +
              " --log-interval " + str(log_interval) +
              " --num-steps " + str(num_steps) +
              " --num-processes " + str(n_cpu) +
              " --lr " + str(learning_rate) +
              " --entropy-coef " + str(entropy_coef) +
              " --value-loss-coef " + str(value_loss_coef) +
              " --ppo-epoch " + str(ppo_epoch) +
              " --num-mini-batch " + str(num_mini_batch) +
              " --num-env-steps " + str(num_env_steps) +


              " --gamma " + str(gamma) +
              " --tau " + str(tau) +
              " --use-linear-lr-decay " +
              " --use-proper-time-limits " +

              " --save-interval " + str(save_model_interval) +

              " --log-dir " + str(log_dir) +
              " --save-dir " + str(save_dir) +
              " --base "  +str(base) +
             # " --no-cuda " +
              other_str

              )

    # if ssh_FLAG:
    #     local_dir = os.path.abspath(group_dir)
    #     remote_dir = AWS_logpath + exp_group_dir + '/'
    #     ssh.upload(local_dir, remote_dir, hostname=hostname, port=port, username=username,
    #                pkey_path=key_path)

    if sync_s3 and bucket_path is not None:
        local_dir = os.path.abspath(group_dir)
        bucket_dir = bucket_path+'/' + local_dir.split('/')[-1]
        cmd = "aws s3 sync {} s3://{} ".format(os.path.abspath(local_dir), bucket_dir)
        os.system(cmd)


