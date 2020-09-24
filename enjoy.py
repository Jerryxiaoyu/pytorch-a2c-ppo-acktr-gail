import sys
path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if path in sys.path:
    sys.path.remove(path)

import argparse
import os

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from utils import LoggerCsv,IO
# workaround to unpickle olf model files
import sys
sys.path.append('a2c_ppo_acktr')

# action_dim = 13
# CPG_enable = 1
# reward_choice= 1
# os.environ["COMMAND_MODE"] = str("FandE")
# os.environ["REWARD_CHOICE"] = str(reward_choice)
# os.environ["ACTION_DIM"] = str(action_dim)
# os.environ["CPG_ENABLE"] = str(CPG_enable)
# os.environ["GLOBAL_CMD"] = 's2-cell6'

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='CellRobotEnvCPG6Traj-v2',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--load-file-dir', default='log-files-SMC/AWS_logfiles/Sep_19_SMC_PPO_RL_Exp18/No_2_CellRobotEnvCPG6Traj-v2_PPO-2020-09-19_14:58:46/model/ppo/CellRobotEnvCPG6Traj-v2_304.pt'   )
parser.add_argument('--result-dir', default=None   )
parser.add_argument('--num-enjoy',type=int, default=1   )
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--data-name',type=str,default=None)
parser.add_argument('--contact-log',type=str,default=None)
parser.add_argument('--no-render', action='store_true', default=False,
                    help='whether to render')
args = parser.parse_args()
num_enjoy = 1
contact_log = args.contact_log

if args.result_dir is None:
    result_dir = 'tmp'
    os.makedirs(result_dir,exist_ok=True)
else:
    result_dir = args.result_dir
if args.data_name is None:
    data_name = '0'
else:
    data_name = args.data_name

logger = LoggerCsv(result_dir, csvname='log_data_{}'.format(data_name))
#logger = None

args.det = not args.non_det

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                            None, None,  device='cpu',
                            allow_early_resets=False)

# Get a render function
if not args.no_render:
    render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
if args.load_file_dir is not None:
    model_path = args.load_file_dir
else:
    model_path = os.path.join(args.load_dir, args.env_name + ".pt")

actor_critic, ob_rms =   torch.load(model_path)

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if (not args.no_render) and render_func is not None:
    render_func('human')

obs = env.reset()

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

velocity_base_lists = []
command_lists = []
reward_lists = []
num_episodes_lists =[]
obs_lists = []
action_list = []
num_episodes = 0
contact_infos =[]

def get_contact_info(env, verbose=False):
    d = env.venv.venv.envs[0].unwrapped.data

    # print(d.ncon)
    geom_list = []
    for coni in range(d.ncon):
        con = d.contact[coni]
        if con.geom1 == 0:
            geom_list.append(con.geom2)
        if verbose:
            print('  Contact %d:' % (coni,))
            print('    dist     = %0.3f' % (con.dist,))
            print('    pos      = %s' % ((con.pos),))
            print('    frame    = %s' % ((con.frame),))
            print('    friction = %s' % ((con.friction),))
            print('    dim      = %d' % (con.dim,))
            print('    geom1    = %d' % (con.geom1,))
            print('    geom2    = %d' % (con.geom2,))

    return geom_list
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, log_info = env.step(action)

    if contact_log is not None:
        contact_info = get_contact_info(env)
        contact_infos.append(contact_info)

    masks.fill_(0.0 if done else 1.0)

    if done:
        num_episodes += 1
    if logger is not None:
        velocity_base_lists.append(log_info[0]['velocity_base'])
        command_lists.append(log_info[0]['commands'])
        reward_lists.append(log_info[0]['rewards'])
        num_episodes_lists.append(num_episodes)
        obs_lists.append(log_info[0]['obs'])
        action_list.append(action.numpy()[0])
    if num_episodes == num_enjoy:
        break

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if (not args.no_render):
        if render_func is not None:
            render_func('human')

if logger is not None:
    velocity_base = np.array(velocity_base_lists, dtype=np.float64)
    commands = np.array(command_lists, dtype=np.float64)
    rewards = np.array(reward_lists, dtype=np.float64)
    num_episodes_lists = np.array(num_episodes_lists, dtype=np.float64).reshape((-1,1))
    obs_lists = np.array(obs_lists, dtype=np.float64)
    action_list = np.array(action_list, dtype=np.float64)

    data = np.concatenate((num_episodes_lists , velocity_base, commands,  obs_lists, rewards, action_list  ), axis=1)

    trajectory = {}
    for j in range(data.shape[0]):
        for i in range(data.shape[1]):
            trajectory[i] = data[j][i]
        logger.log(trajectory)
        logger.write(display=False)
if contact_log is not None:
    contact_log = result_dir
    IO(os.path.join(contact_log , 'log_contact_{}.pkl'.format(data_name))).to_pickle(contact_infos)