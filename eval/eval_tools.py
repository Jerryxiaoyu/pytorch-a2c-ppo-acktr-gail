import os


def find_ExpPath(n, exp_dir_list):
    exp_path = []
    for p in exp_dir_list:
        if '_' in p:
            if p.split('_')[0] == 'No':
                no_exp = int(p.split('_')[1])  # aparse
                if  no_exp == n and p.split('_')[-1] != 'eval':
                    exp_path.append(p)
    if len(exp_path) >1:
        raise Exception("The folder contains more than 2 path of EXP No.{}".format(n))
    if len(exp_path) == 0:
        raise Exception("Cannot find the path of EXP No.{}".format(n))
    return exp_path[0]

def evaluate_fun(result_path,   parms, model_save_num, global_command=None, num_enjoy =1, render = True, monitor = False, rand_init = None ,
                 seed=0, data_name =None, contact_log = None, env_name=None,evaluate_name = "evaluate" ):
    # save_plot_path=os.path.abspath(os.path.join(results_dir,'No_{}-Curve'.format(exp_no)))
    # reward_fun_choice = parms['reward_fun_choice']
    # load_path = os.path.abspath(os.path.join(result_path, 'model/modelmodel'))
    #


    evaluate_path = os.path.join(result_path,evaluate_name)
    os.makedirs(evaluate_path, exist_ok=True)


    exp_group_dir= os.path.abspath(evaluate_path)
    render = render
    store_data = True
    monitor = monitor
    save_model_interval = 1
    gpu_index = 0

    num_threads = 8
    log_interval = 1
    evaluate_flag = True
    max_iter_num = 1  # parms['max_iter_num']max_iter_num = 1#parms['max_iter_num']

    min_batch_size = 2000



    seed = seed
    if env_name is None:
        env_name = parms['env_name']

    if model_save_num  is None:
        model_path = Find_NewestFilePath(os.path.join(result_path, 'model', 'ppo'))
    else:
        model_path = os.path.join(result_path, 'model', 'ppo', env_name + '_{}.pt'.format(model_save_num))

    load_dir = result_path



    os.environ["OMP_NUM_THREADS"] = str(1)

    os.environ["REWARD_CHOICE"] = str(parms['reward_fun_choice'])
    print('REWARD_CHOICE = ', os.getenv('REWARD_CHOICE'))

    if parms['action_dim'] is not None:
        os.environ["ACTION_DIM"] = str(parms['action_dim'])
        print('ACTION_DIM = ', os.getenv('ACTION_DIM'))

    if parms['num_buffer'] is not None:
        os.environ["NUM_BUFFER"] = str(parms['num_buffer'])
        print('NUM_BUFFER = ', os.getenv('NUM_BUFFER'))

    if parms['command_mode'] is not None:
        os.environ["COMMAND_MODE"] = str(parms['command_mode'])
        print('COMMAND_MODE = ', os.getenv('COMMAND_MODE'))

    if parms['buffer_mode'] is not None:
        os.environ["BUFFER_MODE"] = str(parms['buffer_mode'])
        print('BUFFER_MODE = ', os.getenv('BUFFER_MODE'))

    if parms['CPG_enable'] is not None:
        os.environ["CPG_ENABLE"] = str(parms['CPG_enable'])
        print('CPG_ENABLE = ', os.getenv('CPG_ENABLE'))

    if parms['state_mode'] is not None:
        os.environ["STATE_MODE"] = str(parms['state_mode'])
        print('STATE_MODE = ', os.getenv('STATE_MODE'))
    # os.environ["STATE_MODE"] ='vel'
    # print('STATE_MODE = ', os.getenv('STATE_MODE'))
    if 'command_vx_high' in parms.keys():
        if parms['command_vx_high'] is not None:
            os.environ["COMMAND_X"] = str(parms['command_vx_high'])
            print('COMMAND_X = ', os.getenv('COMMAND_X'))

        if parms['command_vy_high'] is not None:
            os.environ["COMMAND_Y"] = str(parms['command_vy_high'])
            print('COMMAND_Y = ', os.getenv('COMMAND_Y'))

        if parms['command_wz_high'] is not None:
            os.environ["COMMAND_Z"] = str(parms['command_wz_high'])
            print('COMMAND_Z = ', os.getenv('COMMAND_Z'))

    if 'vel_filtered' in parms.keys():
        if parms['vel_filtered'] is not None:
            os.environ["VEL_FILTER"] = str(parms['vel_filtered'])
            print('VEL_FILTER = ', os.getenv('VEL_FILTER'))

    if 'turing_flag' in parms.keys():
        if parms['turing_flag'] is not None:
            os.environ["TURING_FLAG"] = str(parms['turing_flag'])
            print('TURING_FLAG = ', os.getenv('TURING_FLAG'))
    if 'xml_name' in parms.keys():
        if parms['xml_name'] is not None:
            os.environ["XML_NAME"] = str(parms['xml_name'])
            print('XML_NAME = ', os.getenv('XML_NAME'))

    if 'sample_mode' in parms.keys():
        if parms['sample_mode'] is not None:
            os.environ["SAMPLE_MODE"] = str(parms['sample_mode'])
            print('SAMPLE_MODE = ', os.getenv('SAMPLE_MODE'))


    if rand_init is None:
        if 'rand_init' in parms.keys():
            if parms['rand_init'] is not None:
                os.environ["RAND_INIT"] = str(parms['RAND_INIT'])
                print('RAND_INIT = ', os.getenv('RAND_INIT'))
    else:
        os.environ["RAND_INIT"] = str(rand_init)
        print('RAND_INIT = ', os.getenv('RAND_INIT'))

    if global_command is not None:

            os.environ["GLOBAL_CMD"] = str(global_command)
            print('GLOBAL_CMD = ', os.getenv('GLOBAL_CMD'))

    #os.environ["XML_NAME"] = "cellrobot_Quadruped_float_limit_ball.xml"

    #os.environ["XML_NAME"] = "cellrobot_Quadruped_float_limit_visualizer_ball2.xml"
    os.environ["XML_NAME"] = "cellrobot_Quadruped_float_limit_visualizer.xml"
    if not render:
        other_str = ' --no-render'
    else:
        other_str = ' '

    if contact_log is not None:
        other_str += ' --contact-log '+str(contact_log)

    os.system("python3 enjoy.py " +
              " --seed " + str(seed) +
              " --env-name " + str(env_name) +
              " --load-dir " + str(load_dir) +
              " --log-interval " + str(log_interval) +
              " --load-file-dir " + str(model_path)+
             " --result-dir " + str(evaluate_path) +
             " --num-enjoy " + str(num_enjoy)+
             " --data-name " +str(data_name) +

              other_str
              # " --gpu-index " + str(gpu_index) +
              # " --store_data " + str(store_data) +
              # " --exp_group_dir " + str(exp_group_dir) +
              # " --monitor " + str(monitor) +
              #   " --evaluate " + str(evaluate_flag)
              )



def Find_NewestFilePath(folderpath, verbose=False):
 lists = os.listdir(folderpath)         # 列出目录的下所有文件和文件夹保存到lists
 lists.sort(key=lambda fn: os.path.getmtime(folderpath + "/" + fn)) # 按时间排序
 file_new = os.path.join(folderpath, lists[-1])      # 获取最新的文件保存到file_new
 if verbose:
  print(file_new)
 return file_new