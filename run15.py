from utils.instrument import VariantGenerator, variant, IO
import os
from datetime import datetime
import shutil
import glob
import paramiko
import time
import utils.ssh as ssh
class VG(VariantGenerator):
    @variant
    def env_name(self):
        return [
            'CellrobotEnvCPG4-v0']  # 'CellrobotEnv-v0' , 'Cellrobot2Env-v0', 'CellrobotSnakeEnv-v0'  , 'CellrobotSnake2Env-v0','CellrobotButterflyEnv-v0', 'CellrobotBigdog2Env-v0'

    @variant
    def seed(self):
        return [123]

    @variant
    def num_steps(self):
        return [2048]

    @variant
    def learning_rate(self):
        return [1e-3]

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
        return [32]

    @variant
    def gamma(self):
        return [0.9985]

    @variant
    def tau(self):
        return [0.995,]

    @variant
    def num_env_steps(self):
        return [5e6]

    ##----------------------------------------------------

    @variant
    def action_dim(self):
        return [2]  # 2,3,13

    @variant
    def reward_fun_choice(self):
        return [30]

    @variant
    def num_buffer(self):
        return [0]

    @variant
    def command_mode(self):
        return ['full']  # full, error, no

    @variant
    def buffer_mode(self):
        return [1]

    @variant
    def CPG_enable(self):
        return [1]

    @variant
    def state_mode(self):
        return ['vel']  # vel , pos, vel_f

    @variant
    def command_vx_high(self):
        return [0.2]

    @variant
    def command_vy_high(self):
        return [0.0]

    @variant
    def command_wz_high(self):
        return [0]  # vel , pos

    @variant
    def vel_filtered(self):
        return [1]

    @variant
    def turing_flag(self):
        return [0]  # 2 line tracking for 5s, 3 for 20s, 1 turning tracking


exp_id = 31
EXP_NAME ='_PPO_RL'
group_note ="************ABOUT THIS EXPERIMENT****************\n" \
            "  " \
        " "

ssh_FLAG = False
AWS_logpath = '/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr/log-files/AWS_logfiles/'
n_cpu =  8
num_threads = 32


# print choices
variants = VG().variants()
num=0
for v in variants:
    num +=1
    print('exp{}: '.format(num), v)

# save gourp parms
exp_group_dir = datetime.now().strftime("%b_%d")+EXP_NAME+'_Exp{}'.format(exp_id)
group_dir = os.path.join('log-files', exp_group_dir)
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


IO('log-files/' + exp_group_dir + '/exp_id{}_param.pkl'.format(exp_id)).to_pickle(param_dict)
print(' Parameters is saved : exp_id{}_param.pkl'.format(exp_id))
# save args prameters
with open(group_dir + '/readme.txt', 'wt') as f:
    print("Welcome to Jerry's lab\n", file=f)
    print(group_note, file=f)


algo ='ppo'

log_interval = 1
save_model_interval = 20

full_output = True
evaluate_monitor = False


# SSH Config
if ssh_FLAG:
    hostname = '2402:f000:6:3801:15f4:4e92:4b69:87da' #'2600:1f16:e7a:a088:805d:16d6:f387:62e5'
    username = 'drl'
    key_path = '/home/ubuntu/.ssh/id_rsa_dl'
    port = 22

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



    os.environ["REWARD_CHOICE"] = str(v['reward_fun_choice'])
    print('export REWARD_CHOICE={}'.format(os.getenv('REWARD_CHOICE')))



    if v['action_dim'] is not None:
        os.environ["ACTION_DIM"] = str(v['action_dim'])
        print('export ACTION_DIM={}'.format(os.getenv('ACTION_DIM')))

    if v['num_buffer'] is not None:
        os.environ["NUM_BUFFER"] = str(v['num_buffer'])
        print('export NUM_BUFFER={}'.format(os.getenv('NUM_BUFFER')))

    if v['command_mode'] is not None:
        os.environ["COMMAND_MODE"] = str(v['command_mode'])
        print('export COMMAND_MODE={}'.format(os.getenv('COMMAND_MODE')))

    if v['buffer_mode'] is not None:
        os.environ["BUFFER_MODE"] = str(v['buffer_mode'])
        print('export BUFFER_MODE={}'.format(os.getenv('BUFFER_MODE')))

    if v['CPG_enable'] is not None:
        os.environ["CPG_ENABLE"] = str(v['CPG_enable'])
        print('export CPG_ENABLE={}'.format(os.getenv('CPG_ENABLE')))

    if v['state_mode'] is not None:
        os.environ["STATE_MODE"] = str(v['state_mode'])
        print('export STATE_MODE={}'.format(os.getenv('STATE_MODE')))

    if v['command_vx_high'] is not None:
        os.environ["COMMAND_X"] = str(v['command_vx_high'])
        print('export COMMAND_X={}'.format(os.getenv('COMMAND_X')))

    if v['command_vy_high'] is not None:
        os.environ["COMMAND_Y"] = str(v['command_vy_high'])
        print('export COMMAND_Y={}'.format(os.getenv('COMMAND_Y')))

    if v['command_wz_high'] is not None:
        os.environ["COMMAND_Z"] = str(v['command_wz_high'])
        print('export COMMAND_Z={}'.format(os.getenv('COMMAND_Z')))

    if v['vel_filtered'] is not None:
            os.environ["VEL_FILTER"] = str(v['vel_filtered'])
            print('export VEL_FILTER={}'.format(os.getenv('VEL_FILTER')))

    if v['turing_flag'] is not None:
            os.environ["TURING_FLAG"] = str(v['turing_flag'])
            print('export TURING_FLAG={}'.format(os.getenv('TURING_FLAG')))


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
              " --save-dir " + str(save_dir)

              )


    print("python3 main.py "  +

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
              " --save-dir " + str(save_dir))
    if ssh_FLAG:
        local_dir = os.path.abspath(group_dir)
        remote_dir = AWS_logpath + exp_group_dir + '/'
        ssh.upload(local_dir, remote_dir, hostname=hostname, port=port, username=username,
                   pkey_path=key_path)


