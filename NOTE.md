

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
    
    ssource activate otter_p36
    
    ```