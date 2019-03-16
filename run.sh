


export ACTION_DIM=13
export NUM_BUFFER=0
export COMMAND_MODE='error'
export BUFFER_MODE=1
export CPG_ENABLE=1
export STATE_MODE='vel'
export REWARD_CHOICE=16

export COMMAND_X=0.5
export COMMAND_Y=0
export COMMAND_Z=0
export VEL_FILTER=0

python3 main.py --env-name "CellrobotEnvCPG4-v0" --algo ppo --use-gae --log-interval 1 --num-steps 4096 --num-processes 8 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.9987 --tau 0.995 --num-env-steps 20000000 --use-linear-lr-decay --use-proper-time-limits --log-dir=logs/cellrobotcpg4/cellrobotcpg4-1 --save-dir=logs/cellrobotcpg4/cellrobotcpg4-1/model --save-interval=50

