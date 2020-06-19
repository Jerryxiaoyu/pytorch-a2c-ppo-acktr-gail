import os
os.chdir('/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr')

# eval/pidcontroller_eval.py --data_name='cons75' --global_command='cons75' --rand_init=1 --seed=17
data_name = 'cons75'
global_command = 'cons75'
rand_init = 1
seed = 12
save_path = '/home/drl/PycharmProjects/rl_baselines/pytorch-a2c-ppo-acktr/test'
# vel tracking -- s1
# gl -- cons175

# for seed in [12,13,14,15,16,17,18]:
#     for global_command in ['cons175']:
#         data_name = global_command+'_'+str(seed)
#         os.system("python3  eval/pidcontroller_eval.py "  +
#
#                       " --data_name " + str(data_name) +
#                       " --global_command " + str(global_command) +
#                   " --save-path " + str(save_path) +
#                       " --rand_init " + str(rand_init) +
#                       " --seed " + str(seed)
#
#                       )


for seed in [12,13,14,15,16,17,18]:
    for global_command in ['cons175']:
        data_name = global_command+'_'+str(seed)
        os.system("python3  eval/pidcontroller_stratight_eval.py "  +

                      " --data_name " + str(data_name) +
                      " --global_command " + str(global_command) +
                  " --save-path " + str(save_path) +
                      " --rand_init " + str(rand_init) +
                      " --seed " + str(seed)

                      )