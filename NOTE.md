

### Log

* 2019.9.18 . baselines-0.1.5 , ShmemVecEnv does not support float64.
   To fix this, I have to change the dict to the following: (in the `baselines/common/vec_env/shmem_vec_env.py` )
   ```buildoutcfg
   _NP_TO_CT = {np.float64: ctypes.c_double,
             np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}
   ```
   
   
   On a server:
   ```buildoutcfg
    cd jerry/otters_pro/
    git clone https://github.com/Jerryxiaoyu/pytorch-a2c-ppo-acktr-gail.git
    cd pytorch-a2c-ppo-acktr-gail
    git checkout run_exps_eval
    
    source activate otter_p36
    
    ```
    
    python main.py  --algo ppo --gail-experts-dir ./gail_experts --gail-batch-size 128 --gail-epoch 5 --lr 0.0003 --eps 1e-05 --alpha 0.99 --entropy-coef 0.01 --value-loss-coef 0.5 --max-grad-norm 0.5 --seed 123 --env-name KinovaReacherJointXYZEnv-v0 --num-env-steps 10000000 --num-processes 8 --num-steps 2048 --ppo-epoch 10 --num-mini-batch 32 --clip-param 0.2 --gae-lambda 0.95 --gamma 0.99 --save-interval 10 --log-interval 10 --log-dir logs-files/20190919-Kinova_Exp5/No_1_KinovaReacherJointXYZEnv-v0_ppo-2019-09-19_163540 --save-dir logs-files/20190919-Kinova_Exp5/No_1_KinovaReacherJointXYZEnv-v0_ppo-2019-09-19_163540/model  --use-gae  --use-proper-time-limits  --use-linear-lr-decay 
 