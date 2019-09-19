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

def Find_NewestFilePath(folderpath, verbose=False):
 lists = os.listdir(folderpath)         # 列出目录的下所有文件和文件夹保存到lists
 lists.sort(key=lambda fn: os.path.getmtime(folderpath + "/" + fn)) # 按时间排序
 file_new = os.path.join(folderpath, lists[-1])      # 获取最新的文件保存到file_new
 if verbose:
  print(file_new)
 return file_new