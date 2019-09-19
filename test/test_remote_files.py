import sys,os
print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import os
from datetime import datetime
import shutil
import glob
import yaml
import time
from  utils  import ssh


remote_FLAG = True
AWS_logpath = '/home/drl/PycharmProjects/JerryRepos/pytorch-a2c-ppo-acktr-gail/logs-files/AWS_logfiles/'



exp_group_dir = "20190919-Kinova_Exp5"
root_path = os.path.abspath( os.path.dirname(__file__))
group_dir = os.path.join(root_path, 'logs-files', exp_group_dir)
#os.makedirs(group_dir)

# SSH Config
if remote_FLAG:
    hostname = 'fe80:0000:6ca2:7e30:9ca3:a287'
    username = 'drl'
    key_path = '/home/ubuntu/.ssh/id_rsa_dl'
    port = 22

if remote_FLAG:
    local_dir = os.path.abspath(group_dir)
    remote_dir = AWS_logpath + exp_group_dir + '/'
    ssh.upload(local_dir, remote_dir, hostname=hostname, port=port, username=username,
               pkey_path=key_path)