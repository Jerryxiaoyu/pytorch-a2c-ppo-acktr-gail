from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly


from gym.envs.registration import registry, register, make, spec

from my_envs.mujoco.cellrobotCPG import CellRobotEnvCPG
from my_envs.mujoco.cellrobotFull import CellRobotEnvFull
from my_envs.mujoco.cellrobotCPG2 import CellRobotEnvCPG2
from my_envs.mujoco.cellrobotCPG3 import CellRobotEnvCPG3
from my_envs.mujoco.cellrobotCPG4 import CellRobotEnvCPG4
from my_envs.mujoco.cellrobotCPG5_SMC import CellRobotEnvCPG5
from my_envs.mujoco.cellrobotCPG6_goal_SMC import CellRobotEnvCPG6Goal
from my_envs.mujoco.cellrobotCPG6_goal_SMC import CellRobotEnvCPG6GoalTraj
from my_envs.mujoco.cellrobotCPG6_goal_points import CellRobotEnvCPG6Target
from my_envs.mujoco.cellrobotCPG6_goal_points import CellRobotEnvCPG6NewTarget, \
    CellRobotEnvCPG6NewEVALTarget, CellRobotEnvCPG6NewMultiTarget, CellRobotEnvCPG6NewEVALTargetILC
from my_envs.mujoco.my_ant import MyAntEnv

register(
    id='CellrobotEnvFull-v0',
    entry_point='my_envs.mujoco:CellRobotEnvFull ',
    max_episode_steps=2000,
    reward_threshold=6000.0,
)


register(
    id='CellrobotEnvCPG-v0',
    entry_point='my_envs.mujoco:CellRobotEnvCPG ',
    max_episode_steps=2000,
    reward_threshold=6000.0,
)
register(
    id='CellrobotEnvCPG2-v0',
    entry_point='my_envs.mujoco:CellRobotEnvCPG2 ',
    max_episode_steps=2000,
    reward_threshold=6000.0,
)
register(
    id='CellrobotEnvCPG3-v0',
    entry_point='my_envs.mujoco:CellRobotEnvCPG3 ',
    max_episode_steps=20,
    reward_threshold=6000.0,
)

register(
    id='CellrobotEnvCPG4-v0',
    entry_point='my_envs.mujoco:CellRobotEnvCPG4 ',
    max_episode_steps=2000,
    reward_threshold=6000.0,
)

register(
    id='CellrobotEnvCPG4-v1',
    entry_point='my_envs.mujoco:CellRobotEnvCPG4 ',
    max_episode_steps=4000,
    reward_threshold=6000.0,
)
register(
    id='MyAnt-v2',
    entry_point='my_envs.mujoco:MyAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)



register(
    id='CellrobotEnvCPG5-v0',
    entry_point='my_envs.mujoco:CellRobotEnvCPG5',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=0)
)

register(
    id='CellrobotEnvCPG5-v1',
    entry_point='my_envs.mujoco:CellRobotEnvCPG5',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=1)
)


register(
    id='CellRobotEnvCPG6Goal-v1',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6Goal',
    max_episode_steps=2000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=1)
)

register(
    id='CellRobotEnvCPG6Goal-v2',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6Goal',
    max_episode_steps=2000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2)
)



register(
    id='CellRobotEnvCPG6Traj-v1',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6GoalTraj',
    max_episode_steps=2000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=1)
)

register(
    id='CellRobotEnvCPG6Traj-v2',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6GoalTraj',
    max_episode_steps=2000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2)
)

register(
    id='CellRobotEnvCPG6Traj-v3',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6GoalTraj',
    max_episode_steps=2000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2,
                 isRootposNotInObs = True

                 )
)

register(
    id='CellRobotEnvCPG6Target-v2',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6Target',
    max_episode_steps=2000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2,
                 max_steps = 400,

                trajectory_length = 40,

                 robot_state_dim = 41,
                isRenderGoal = 1,
                 )
)

register(
    id='CellRobotEnvCPG6Target-v3',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6Target',
    max_episode_steps=2000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2,
                 max_steps = 400,

                trajectory_length = 0,
                 robot_state_dim = 41,
                isRenderGoal = 1,
                 )
)


register(
    id='CellRobotEnvCPG6NewTarget-v2',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6NewTarget',
    max_episode_steps=2000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2,
                 max_steps = 400,

                trajectory_length = 40,

                 robot_state_dim = 42,
                isRenderGoal = 0,
                 sample_mode = 1
                 )
)


register(
    id='CellRobotEnvCPG6NewMultiTarget-v2',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6NewMultiTarget',
    max_episode_steps=2000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2,
                 max_steps = 200,

                hardReset_per_reset= 50,

                num_goals = 5,

                trajectory_length = 40,

                 robot_state_dim = 42,
                 isRenderGoal = 1,
                 sample_mode = 1
                 )
)

register(
    id='CellRobotEnvCPG6NewMultiTarget-v3',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6NewMultiTarget',
    max_episode_steps=2000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2,
                 max_steps = 80,

                hardReset_per_reset= 50,

                num_goals = 2,

                trajectory_length = 40,

                 robot_state_dim = 42,
                 isRenderGoal = 1,
                 sample_mode = 1
                 )
)


register(
    id='CellRobotEnvCPG6NewTargetEVAL-v2',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6NewEVALTarget',
    max_episode_steps=10000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2,
                 max_steps = 45,
                 hardReset_per_reset= 500,

                 trajectory_length = 40,
                 robot_state_dim = 42,
                 isRenderGoal = 1,
                 sample_mode = 1,
                 )
)

register(
    id='CellRobotEnvCPG6NewTargetEVAL-v3',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6NewEVALTarget',
    max_episode_steps=10000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2,
                 max_steps = 1,
                 hardReset_per_reset= 50000,
                render_traj_mode= 5,

                 trajectory_length = 40,
                 robot_state_dim = 42,
                 isRenderGoal = 1,
                 sample_mode = 1,
                 )
)

register(
    id='CellRobotEnvCPG6NewTargetEVAL-ILC-v2',
    entry_point='my_envs.mujoco:CellRobotEnvCPG6NewEVALTargetILC',
    max_episode_steps=10000,
    reward_threshold=6000.0,
    kwargs=dict( control_skip = 5,
                 cpg_mode=2,
                 max_steps = 45,
                 hardReset_per_reset= 500,

                 trajectory_length = 40,
                 robot_state_dim = 42,
                 isRenderGoal = 1,
                 sample_mode = 1,
                 )
)


#
#
#
# register(
#     id='CellRobotEnvCPG6Target-v3',
#     entry_point='my_envs.mujoco:CellRobotEnvCPG6Target',
#     max_episode_steps=2000,
#     reward_threshold=6000.0,
#     kwargs=dict( control_skip = 5,
#                  cpg_mode=2,
#                  max_steps = 2000,
#
#
#                  robot_state_dim = 41,
#                 isRenderGoal = 0,
#                  )
# )