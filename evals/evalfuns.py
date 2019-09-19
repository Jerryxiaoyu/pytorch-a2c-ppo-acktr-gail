import numpy as np
import os
from matplotlib.pylab import plt
from baselines.common import plot_util as pu
from evals.utils import Find_NewestFilePath


def evaluate_fun(result_path, parms, config  ):


    evaluate_path = os.path.join(result_path,'evaluate')
    os.makedirs(evaluate_path, exist_ok=True)

    exp_group_dir= os.path.abspath(evaluate_path)

    log_interval = config['log_interval']

    model_save_num = config['model_save_num']

    seed = config['seed']
    env_name = parms['env_name']
    if model_save_num  is None:
        # TODO check 'ppo' 'sca'
        model_path = Find_NewestFilePath(os.path.join(result_path, 'model', 'ppo'))
    else:
        model_path = os.path.join(result_path, 'model', 'ppo', env_name + '.{}.pt'.format(model_save_num))

    load_name = model_path

    others_str = [" "]
    if config['isRender']:
        others_str.append(" --isRender ")

    others_str = ''.join(others_str)

    os.system("python enjoy.py " +
              " --seed " + str(seed) +
              " --env-name " + str(env_name) +


              " --load-dir " + str(load_name) +
              " --log-interval " + str(log_interval)

            + others_str

              )


def plot_learning_curve(r, save_plot_path=None, exp_no=1, figsize=(8,6)):

    name = 'No.{} learning curve'.format(exp_no)
    #plt.plot(r.progress.TimestepsSoFar, r.progress.EpRewMean)
    fig, ax = pu.plot_results(r, average_group=True, split_fn=lambda _: '', shaded_std=False, figsize=figsize)

    plt.title(name)
    if save_plot_path is not None:
        plt.savefig( save_plot_path+'-learning_curve.jpg' )
    else:
        plt.show()


