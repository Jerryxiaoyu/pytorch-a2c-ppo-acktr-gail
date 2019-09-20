from utils.instrument import VariantGenerator, variant, IO
import os
from datetime import datetime
import shutil
import glob
import yaml
import time
import utils.ssh as ssh
class VG(VariantGenerator):
    @variant
    def seed(self): #random seed (default: 1)
        return [123]
    @variant
    def env_name(self):  #environment to train on (default: PongNoFrameskip-v4) #'KinovaReacherTorqueXYZEnv-v0',
        return ['KinovaReacherXYZ-v0' ,]# 'KinovaReacherXYZ-v0','KinovaReacherTorqueXYZEnv-v0'
    @variant
    def algo(self):
        return ['ppo']
    @variant
    def learning_rate(self): # learning rate (default: 7e-4)
        return [3e-4]
    @variant
    def alpha(self): # 'RMSprop optimizer apha (default: 0.99)'
        return [0.99]
    @variant
    def eps(self): # 'RMSprop optimizer epsilon (default: 1e-5)'
        return [1e-5]
    @variant
    def num_steps(self): #number of forward steps in A2C (default: 5)
        return [2048 ]
    @variant
    def gail_batch_size(self): #'gail batch size (default: 128)'
        return [128]
    @variant
    def gail_experts_dir(self): #'directory that contains expert demonstrations for gail'
        return ['./gail_experts']
    @variant
    def gail_epoch(self): # 'gail epochs (default: 5)'
        return [5]
    @variant
    def entropy_coef(self): #'entropy term coefficient (default: 0.01)'
        return [0.01 ]
    @variant
    def value_loss_coef(self):#'value loss coefficient (default: 0.5)'
        return [0.5]
    @variant
    def ppo_epoch(self): # number of ppo epochs (default: 4)
        return [10]
    @variant
    def num_mini_batch(self): #number of batches for ppo (default: 32)
        return [32]
    @variant
    def gamma(self): # 'discount factor for rewards (default: 0.99)'
        return [0.99]
    @variant
    def gae_lambda(self): # 'gae lambda parameter (default: 0.95)'
        return [0.95]
    @variant
    def num_env_steps(self): #number of environment steps to train (default: 10e6)
        return [1e7]
    @variant
    def max_grad_norm(self): # max norm of gradients (default: 0.5)
        return [0.5]
    @variant
    def num_processes(self): # CPU processes to use (default: 16)
        return [36]
    @variant
    def clip_param(self):  # ppo clip parameter (default: 0.2)
        return [0.2]
    @variant
    def use_gae_flag(self):  # use generalized advantage estimation
        return [True]
    @variant
    def no_cuda_flag(self):  # disables CUDA training
        return [False]
    @variant
    def use_proper_time_limits_flag(self):  # compute returns taking into account time limits
        return [True]
    @variant
    def recurrent_policy_flag(self):  # use a recurrent policy
        return [False]
    @variant
    def use_linear_lr_decay_flag(self):  # use a linear schedule on the learning rate
        return [True]

    @variant
    def gail_flag(self):  # do imitation learning with gail
        return [False]
 ##----------------------------------------------------
    @variant
    def reward_fun_choice(self):
        return [ 0,1,2,3] #,1,2,3

exp_id = 9
remote_FLAG = False
EXP_NAME ='Kinova'
group_note ="************ABOUT THIS EXPERIMENT****************\n" \
            "  " \
          " "


AWS_logpath = '/home/drl/PycharmProjects/JerryRepos/pytorch-a2c-ppo-acktr-gail/logs-files/AWS_logfiles/'


# print choices
variants = VG().variants()
num=0
for v in variants:
    num +=1
    print('exp{}: '.format(num), v)

# save gourp parms
exp_group_dir = datetime.now().strftime("%Y%m%d-")+EXP_NAME+'_Exp{}'.format(exp_id)
group_dir = os.path.join('logs-files', exp_group_dir)
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

print(param_dict)
with open('logs-files/' + exp_group_dir + '/exp_id{}_param.yaml'.format(exp_id), 'w') as f:
    yaml.dump(param_dict, f)

print(' Parameters is saved : exp_id{}_param.yaml'.format(exp_id))
# save args prameters
with open(group_dir + '/readme.txt', 'wt') as f:
    print("Welcome to Jerry's lab\n", file=f)
    print(group_note, file=f)

log_interval = 10
save_model_interval = 10


full_output = True
evaluate_monitor = False

# SSH Config
if remote_FLAG:
    hostname = 'fe80::6ca2:7e30:9ca3:a287'
    username = 'drl'
    key_path = '/home/ubuntu/.ssh/id_rsa_dl'
    port = 22

# run
num_exp =0
for v in variants:
    num_exp += 1
    print(v)
    time_now = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(time.time()))
    # load parm
    exp_name = 'No_{}_{}_{}-{}'.format(num_exp, v['env_name'],v['algo'],   time_now)
    log_dir =  os.path.join(group_dir,exp_name)
    save_dir =  os.path.join(log_dir,'model')
    seed = v['seed']
    env_name = v['env_name']

    algo =  v['algo']
    gamma = v['gamma']
    gae_lambda = v['gae_lambda']
    n_cpu = v['num_processes']

    learning_rate = v['learning_rate']
    entropy_coef = v['entropy_coef']
    value_loss_coef = v['value_loss_coef']
    ppo_epoch = v['ppo_epoch']
    num_env_steps = int(v['num_env_steps'])
    num_mini_batch = v['num_mini_batch']
    gail_experts_dir = v['gail_experts_dir']
    gail_batch_size = v['gail_batch_size']
    gail_epoch = v['gail_epoch']
    eps = v['eps']
    alpha = v['alpha']
    max_grad_norm = v['max_grad_norm']
    num_steps = v['num_steps']
    clip_param = v['clip_param']


    others_str = [" "]
    if v['gail_flag']:
        others_str.append(" --gail ")
    if v['use_gae_flag']:
        others_str.append(" --use-gae ")
    if v['no_cuda_flag']:
        others_str.append(" --no-cuda ")
    if v['use_proper_time_limits_flag']:
        others_str.append(" --use-proper-time-limits ")
    if v['recurrent_policy_flag']:
        others_str.append(" --recurrent-policy ")
    if v['use_linear_lr_decay_flag']:
        others_str.append(" --use-linear-lr-decay ")

    others_str = ''.join(others_str)


    #--------custom enviornment variables---------------
    os.environ["REWARD_CHOICE"] = str(v['reward_fun_choice'])
    print('REWARD_CHOICE = ', os.getenv('REWARD_CHOICE'))


#os.system
    os.system("python main.py "  +

              " --algo " + str(algo) +

              " --gail-experts-dir " + str(gail_experts_dir) +
              " --gail-batch-size " + str(gail_batch_size) +
              " --gail-epoch "      + str(gail_epoch) +

              " --lr " + str(learning_rate) +
              " --eps "+ str(eps) +
              " --alpha "+ str(alpha) +

              " --entropy-coef " + str(entropy_coef) +
              " --value-loss-coef " + str(value_loss_coef) +

              " --max-grad-norm " + str(max_grad_norm) +

              " --seed " + str(seed) +
              " --env-name " + str(env_name) +
              " --num-env-steps " + str(num_env_steps) +
              " --num-processes " + str(n_cpu) +

              " --num-steps " + str(num_steps) +

              " --ppo-epoch " + str(ppo_epoch) +
              " --num-mini-batch " + str(num_mini_batch) +
              " --clip-param " + str(clip_param) +
              " --gae-lambda " + str(gae_lambda) +
              " --gamma " + str(gamma) +

              " --save-interval " + str(save_model_interval) +
             # " --eval-interval " + str(eval_interval) +
              " --log-interval " + str(log_interval) +

              " --log-dir " + str(log_dir) +
              " --save-dir " + str(save_dir)+

              others_str
              )

    if remote_FLAG:
        local_dir = os.path.abspath(group_dir)
        remote_dir = AWS_logpath + exp_group_dir + '/'
        ssh.upload(local_dir, remote_dir, hostname=hostname, port=port, username=username,
                   pkey_path=key_path)


