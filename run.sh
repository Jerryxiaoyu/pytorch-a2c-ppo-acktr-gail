
# 32
export REWARD_CHOICE=32
export ACTION_DIM=2
export NUM_BUFFER=0
export COMMAND_MODE=full
export BUFFER_MODE=1
export CPG_ENABLE=1
export STATE_MODE=pos
export COMMAND_X=1
export COMMAND_Y=0.0
export COMMAND_Z=0
export VEL_FILTER=1
export TURING_FLAG=1
export RAND_INIT=0
python3 main.py  --env-name CellrobotEnvCPG4-v0 --algo ppo --use-gae  --log-interval 1 --num-steps 2048 --num-processes 8 --lr 0.001 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --num-env-steps 10000000 --gamma 0.9985 --tau 0.995 --use-linear-lr-decay  --use-proper-time-limits  --save-interval 20 --log-dir log-files/Mar_18_PPO_RL_Exp32/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-18_13:52:13 --save-dir log-files/Mar_18_PPO_RL_Exp32/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-18_13:52:13/model


# 33

export REWARD_CHOICE=32
export ACTION_DIM=2
export NUM_BUFFER=0
export COMMAND_MODE=full
export BUFFER_MODE=1
export CPG_ENABLE=1
export STATE_MODE=pos
export COMMAND_X=1
export COMMAND_Y=0.0
export COMMAND_Z=0
export VEL_FILTER=1
export TURING_FLAG=1
export RAND_INIT=1
python3 main.py  --env-name CellrobotEnvCPG4-v1 --algo ppo --use-gae  --log-interval 1 --num-steps 2048 --num-processes 8 --lr 0.001 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --num-env-steps 10000000 --gamma 0.9985 --tau 0.995 --use-linear-lr-decay  --use-proper-time-limits  --save-interval 20 --log-dir log-files/Mar_18_PPO_RL_Exp33/No_1_CellrobotEnvCPG4-v1_PPO-2019-03-18_13:54:11 --save-dir log-files/Mar_18_PPO_RL_Exp33/No_1_CellrobotEnvCPG4-v1_PPO-2019-03-18_13:54:11/model


# 34

export REWARD_CHOICE=32
export ACTION_DIM=2
export NUM_BUFFER=0
export COMMAND_MODE=full
export BUFFER_MODE=1
export CPG_ENABLE=1
export STATE_MODE=vel
export COMMAND_X=1
export COMMAND_Y=0.0
export COMMAND_Z=0
export VEL_FILTER=1
export TURING_FLAG=1
export RAND_INIT=0
python3 main.py  --env-name CellrobotEnvCPG4-v0 --algo ppo --use-gae  --log-interval 1 --num-steps 2048 --num-processes 8 --lr 0.001 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --num-env-steps 10000000 --gamma 0.9985 --tau 0.995 --use-linear-lr-decay  --use-proper-time-limits  --save-interval 20 --log-dir log-files/Mar_18_PPO_RL_Exp34/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-18_13:54:47 --save-dir log-files/Mar_18_PPO_RL_Exp34/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-18_13:54:47/model


python3 eval/evaluate.py --env_name='CellrobotEnvCPG4-v1' --group_dir='log-files/Mar_22_PPO_RL_Exp42' --exp_id=42 --data_name='c3' --global_command='c3' --rand_init=0 --seed=17
