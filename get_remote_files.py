import sys
import os
import argparse
import ast


exp_dir = "20190920-Kinova_Exp9"

remote_ip = "18.216.24.207"

remote_root = "/home/ubuntu/jerry/otters_pro/pytorch-a2c-ppo-acktr-gail/logs-files"

remote_path = os.path.join(remote_root, exp_dir)

local_path = "/home/drl/PycharmProjects/JerryRepos/pytorch-a2c-ppo-acktr-gail/logs-files/AWS_logfiles"

local_exp_path = os.path.join(local_path, exp_dir)
if not os.path.exists(local_exp_path):
    os.mkdir(local_exp_path)

os.system("scp -r -i ~/.ssh/aws_ohio.pem " +
          "ubuntu@{}".format(remote_ip)+
          ":{}".format(remote_path)+
          " {}".format(local_exp_path))

