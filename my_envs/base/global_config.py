class global_var:
    curriculum = 0.3
    iter_rl = 0


def set_curriculum(param):
    global_var.curriculum = param
    return global_var.curriculum

def get_curriculum():
    return global_var.curriculum


def set_iter_rl(param):
    global_var.iter_rl = param
    return global_var.iter_rl

def get_iter_rl():
    return global_var.iter_rl