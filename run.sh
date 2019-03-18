
# No 24
export REWARD_CHOICE=30
export ACTION_DIM=2
export NUM_BUFFER=0
export COMMAND_MODE=full
export BUFFER_MODE=1
export CPG_ENABLE=1
export STATE_MODE=vel
export COMMAND_X=0.2
export COMMAND_Y=0
export COMMAND_Z=0
export VEL_FILTER=1
export TURING_FLAG=0
python3 main.py  --env-name CellrobotEnvCPG4-v0 --algo ppo --use-gae  --log-interval 1 --num-steps 2048 --num-processes 8 --lr 0.001 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --num-env-steps 5000000 --gamma 0.9985 --tau 0.95 --use-linear-lr-decay  --use-proper-time-limits  --save-interval 20 --log-dir log-files/Mar_17_PPO_RL_Exp24/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-17_20:47:25 --save-dir log-files/Mar_17_PPO_RL_Exp24/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-17_20:47:25/model

# No 25
export REWARD_CHOICE=30
export ACTION_DIM=2
export NUM_BUFFER=0
export COMMAND_MODE=full
export BUFFER_MODE=1
export CPG_ENABLE=1
export STATE_MODE=vel
export COMMAND_X=0.2
export COMMAND_Y=0
export COMMAND_Z=0
export VEL_FILTER=0
export TURING_FLAG=0
python3 main.py  --env-name CellrobotEnvCPG4-v0 --algo ppo --use-gae  --log-interval 1 --num-steps 2048 --num-processes 8 --lr 0.001 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --num-env-steps 5000000 --gamma 0.9985 --tau 0.995 --use-linear-lr-decay  --use-proper-time-limits  --save-interval 20 --log-dir log-files/Mar_17_PPO_RL_Exp25/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-17_20:49:51 --save-dir log-files/Mar_17_PPO_RL_Exp25/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-17_20:49:51/model

No 29
export REWARD_CHOICE=30
export ACTION_DIM=2
export NUM_BUFFER=2
export COMMAND_MODE=full
export BUFFER_MODE=1
export CPG_ENABLE=1
export STATE_MODE=vel
export COMMAND_X=0.2
export COMMAND_Y=0.0
export COMMAND_Z=0
export VEL_FILTER=1
export TURING_FLAG=0
python3 main.py  --env-name CellrobotEnvCPG4-v0 --algo ppo --use-gae  --log-interval 1 --num-steps 2048 --num-processes 8 --lr 0.001 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --num-env-steps 10000000 --gamma 0.9985 --tau 0.995 --use-linear-lr-decay  --use-proper-time-limits  --save-interval 20 --log-dir log-files/Mar_17_PPO_RL_Exp29/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-17_21:22:53 --save-dir log-files/Mar_17_PPO_RL_Exp29/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-17_21:22:53/model
