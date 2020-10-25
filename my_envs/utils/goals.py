import numpy as np
import transformations
from scipy.interpolate import interp1d



def generate_same_interval_eight_curve(A=6, b=2, N= 10000, dis= 0.3):
    DIS = dis#0.3
    A = A
    b = b
    N = N
    t = np.linspace(0, 2 * np.pi, num=N)
    x = A * np.sin(b * t)
    y = A * np.sin(b * t) * np.cos(b * t)
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()

    points = []
    points.append(xy[0])

    N = xy.shape[0]
    idx = 1

    def find_nearPoint(idx):
        for try_idx in range(idx, N):
            dis = np.linalg.norm(xy[try_idx] - xy[idx])

            if dis > DIS - 0.01 and dis < DIS + 0.01:
                points.append(xy[try_idx])
                idx = try_idx
                return idx
        return None

    while idx < N:
        idx = find_nearPoint(idx)
        if idx is None:
            break

    points = np.array(points)
    return points






def generate_circle_curve(R= 3, direction=1, vel=0.1, dt = 0.05, least_N = 4000, no_extend=False):
    theta = np.pi * 2

    dt = 0.05
    T = R * theta / vel
    num_N = int(T / dt)

    if direction == 1:
        t = np.linspace(-np.pi / 2, -np.pi / 2 + theta, num=num_N, endpoint=True)
        x = 0 + R * np.cos(t)
        y = R + R * np.sin(t)
    else:
        t = np.linspace(np.pi / 2 - theta, np.pi / 2, num=num_N, endpoint=True)
        x = 0 + R * np.cos(t)
        y = -R + R * np.sin(t)
        x = x[::-1]
        y = y[::-1]

    xy = np.concatenate((x, y )).reshape((2, -1)).transpose()


    if not no_extend:
        num_xy = xy.shape[0]
        cnt = int(np.ceil(least_N * 1.5 / num_xy))

        tmp = xy
        if cnt > 1:
            for _ in range(cnt):
                tmp = np.concatenate([tmp, xy], axis=0)
    else:
        tmp = xy
        tmp = np.concatenate([tmp, xy], axis=0)



    return tmp





def generate_eight_curve(A= 6, b=2, vel=0.1, dt = 0.05, least_N = 4000, no_extend=False):
    A = A
    b = b
    N = 20000
    t = np.linspace(0, np.pi, num=N)
    x = A * np.sin(b * t)
    y = A * np.sin(b * t) * np.cos(b * t)
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()

    sum_traj = np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum()
    new_N = sum_traj / (vel * dt) * 1.1


    t = np.linspace(0, np.pi, num=new_N)
    x = A * np.sin(b * t)
    y = A * np.sin(b * t) * np.cos(b * t)
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()

    num_xy = xy.shape[0]
    cnt = int(np.ceil(least_N*1.5/num_xy))

    # tmp = xy
    # if cnt >1:
    #     for _ in range(cnt):
    #         tmp = np.concatenate([tmp, xy], axis=0)

    if not no_extend:
        num_xy = xy.shape[0]
        cnt = int(np.ceil(least_N * 1.5 / num_xy))

        tmp = xy
        if cnt > 1:
            for _ in range(cnt):
                tmp = np.concatenate([tmp, xy], axis=0)
    else:
        tmp = xy
        tmp = np.concatenate([tmp, xy], axis=0)

    return tmp


def get_same_interrval(xy, vel=0.1, Dt=0.05):
    points = []
    points.append(xy[0])

    DIS = vel * Dt

    end_index = 0
    try_index = 0
    last_index = 0
    while end_index < (xy.shape[0] - 10):
        # print(end_index)
        sum_dis = 0
        for end_index in range(try_index + 1, xy.shape[0]):

            dis = np.linalg.norm(xy[end_index] - xy[end_index - 1])

            sum_dis += dis

            if sum_dis < DIS:
                end_index += 1
            else:
                last_index = try_index
                points.append(xy[end_index])
                try_index = end_index
                sum_dis = 0
                break

    points = np.array(points)
    return points

def generate_star_curve(A= 0.5, C= 1,  vel=0.1, dt = 0.05, least_N = 4000, no_extend=False):

    N = 20000
    t = np.linspace(0, 4 * np.pi, num=N)
    x = C * 3 * np.sin(t) / 2 + C * A * np.sin(3 * t / 2)
    y = C * 3 * np.cos(t) / 2 - C * A * np.cos(3 * t / 2)

    xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    #xy = get_same_interrval(xy, vel=vel, Dt=0.05)

    sum_traj = np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum()
    new_N = sum_traj / (vel * dt) * 1.1

    t = np.linspace(0, 4 * np.pi, num=new_N)
    x = C * 3 * np.sin(t) / 2 + C * A * np.sin(3 * t / 2)
    y = C * 3 * np.cos(t) / 2 - C * A * np.cos(3 * t / 2)
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    xy = get_same_interrval(xy, vel=vel, Dt=0.05)

    num_xy = xy.shape[0]
    cnt = int(np.ceil(least_N*1.5/num_xy))

    # tmp = xy
    # if cnt >1:
    #     for _ in range(cnt):
    #         tmp = np.concatenate([tmp, xy], axis=0)

    if not no_extend:
        num_xy = xy.shape[0]
        cnt = int(np.ceil(least_N * 1.5 / num_xy))

        tmp = xy
        if cnt > 1:
            for _ in range(cnt):
                tmp = np.concatenate([tmp, xy], axis=0)
    else:
        tmp = xy
        tmp = np.concatenate([tmp, xy], axis=0)

    return tmp

def generate_heart_curve(A= 0.5, C= 1,  vel=0.1, dt = 0.05, least_N = 4000, no_extend=False):
    A = 0.1
    C = 1
    N = 20000
    t = np.linspace(0, 2 * np.pi, num=N)
    x = A * (16 * np.sin(t) ** 3)
    y = A * (15 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()

    sum_traj = np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum()
    new_N = sum_traj / (vel * dt) * 1.1

    t = np.linspace(0, 2 * np.pi, num=new_N)
    x = A * (16 * np.sin(t) ** 3)
    y = A * (15 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()

    num_xy = xy.shape[0]
    cnt = int(np.ceil(least_N*1.5/num_xy))

    # tmp = xy
    # if cnt >1:
    #     for _ in range(cnt):
    #         tmp = np.concatenate([tmp, xy], axis=0)

    if not no_extend:
        num_xy = xy.shape[0]
        cnt = int(np.ceil(least_N * 1.5 / num_xy))

        tmp = xy
        if cnt > 1:
            for _ in range(cnt):
                tmp = np.concatenate([tmp, xy], axis=0)
    else:
        tmp = xy
        tmp = np.concatenate([tmp, xy], axis=0)

    return tmp


def generate_rect_curve( vel=0.1, dt = 0.05, least_N = 4000, no_extend=False):

    N = 1450
    xy = cal_rectangle(N )

    sum_traj = np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum()
    new_N = sum_traj / (vel * dt) * 1.1



    xy = cal_rectangle(int(N/14) )

    num_xy = xy.shape[0]
    cnt = int(np.ceil(least_N*1.5/num_xy))

    # tmp = xy
    # if cnt >1:
    #     for _ in range(cnt):
    #         tmp = np.concatenate([tmp, xy], axis=0)

    if not no_extend:
        num_xy = xy.shape[0]
        cnt = int(np.ceil(least_N * 1.5 / num_xy))

        tmp = xy
        if cnt > 1:
            for _ in range(cnt):
                tmp = np.concatenate([tmp, xy], axis=0)
    else:
        tmp = xy
        tmp = np.concatenate([tmp, xy], axis=0)

    return tmp

def generate_butterfly_curve(vel=0.2, dt = 0.05, least_N = 4000):
    data = np.loadtxt('data/contours/butterfly.txt')

    xy = data[40000:] * 50

    sum_traj = np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum()
    new_N = int(sum_traj / (vel * dt) * 1.1)

    interval = int(xy.shape[0] / new_N)
    idxs = np.clip(np.arange(0, xy.shape[0], interval), 0, xy.shape[0] - 1)

    xy = xy[idxs]

    num_xy = xy.shape[0]
    cnt = int(np.ceil(least_N * 1.5 / num_xy))

    tmp = xy
    if cnt > 1:
        for _ in range(cnt):
            tmp = np.concatenate([tmp, xy], axis=0)
    return tmp

def get_arc_data(mode,x0,y0, N, R= 1):
    if mode =='1':
        t = np.linspace(0, np.pi / 2, num=N, endpoint=True)
        x = x0+ 0 + R * np.cos(t)
        y = y0+ -R + R * np.sin(t)
        x = x[::-1]
        y = y[::-1]
        xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    elif mode =='2':
        t = np.linspace(-np.pi/2,  0 , num=N, endpoint=True)
        x = x0+-R + R * np.cos(t)
        y = y0+0 + R * np.sin(t)
        x = x[::-1]
        y = y[::-1]
        xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    elif mode =='3':
        t = np.linspace(np.pi  , np.pi/2*3 , num=N, endpoint=True)
        x = x0+0 + R * np.cos(t)
        y = y0+R + R * np.sin(t)
        x = x[::-1]
        y = y[::-1]
        xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    elif mode =='4':
        t = np.linspace(np.pi/2  , np.pi , num=N, endpoint=True)
        x = x0+R + R * np.cos(t)
        y = y0+0 + R * np.sin(t)
        x = x[::-1]
        y = y[::-1]
        xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    return xy

def cal_rectangle(N ):
    L = 1
    #N = 1450
    # 8+ int(L/2*1.5707*4)
    points = []

    x = np.linspace(0, L / 2, num=N)
    y = np.zeros_like(x)
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    points.append(xy)

    xy = get_arc_data('1', x[-1], y[-1], int(L / 2 * 1.5707 * N))
    points.append(xy)

    y = np.linspace(xy[-1, 1], xy[-1, 1] - L, num=2 * N)
    x = np.ones_like(y) * xy[-1, 0]
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    points.append(xy)

    xy = get_arc_data('2', x[-1], y[-1], int(L / 2 * 1.5707 * N))
    points.append(xy)

    x = np.linspace(xy[-1, 0], xy[-1, 0] - L, num=2 * N)
    y = np.ones_like(x) * xy[-1, 1]
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    points.append(xy)

    xy = get_arc_data('3', x[-1], y[-1], int(L / 2 * 1.5707 * N))
    points.append(xy)

    y = np.linspace(xy[-1, 1], xy[-1, 1] + L, num=2 * N)
    x = np.ones_like(y) * xy[-1, 0]
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    points.append(xy)

    xy = get_arc_data('4', x[-1], y[-1], int(L / 2 * 1.5707 * N))
    points.append(xy)

    x = np.linspace(xy[-1, 0], xy[-1, 0] + L / 2, num=N)
    y = np.ones_like(x) * xy[-1, 1]
    xy = np.concatenate((x[None], y[None]), axis=0).transpose()
    points.append(xy)

    for i in range(len(points)):
        if i == 0:
            xy = points[0]
        else:
            xy = np.concatenate((xy, points[i]), axis=0)

    return xy
def generate_point_in_arc_area(center_p, norm_dir,  theta= np.deg2rad(30), dis_range= (30, 100), side = np.random.choice(2) ):
    """
    生成一个点，在给扇形区域内, 在xz平面内。
    :param center_p: 当前点位置
    :param norm_dir: 面朝的方向
    :param theta:  扇形区域的夹角的一半，
    :param dis_range: 随机距离
    :return: [x,y,z]
    """
    center_p = np.array(center_p)
    norm_dir = np.array(norm_dir)

    dis = np.random.uniform(dis_range[0], dis_range[1])

    point = center_p + norm_dir *dis
    left_side_vec = np.cross(np.array([0,1,0]), norm_dir)

    Rot_T = transformations.rotation_matrix(np.pi, [0,1,0])
    right_side_vec = Rot_T[0:3,0:3].dot(left_side_vec)

    theta = np.random.uniform(0, theta)
    side_dis = np.random.uniform(0, dis*np.tan(theta))

    dir_choice = side
    final_p = (1-dir_choice)*side_dis*left_side_vec + dir_choice*side_dis*right_side_vec + point

    return final_p


def sample_traj_spline(root_position,norm_dir,
                       T=10,
                       rand_init=20,
                       theta_list =  [np.deg2rad(60),np.deg2rad(60),np.deg2rad(60),np.deg2rad(60),np.deg2rad(60)  ],
                       dis_list   = [(20,50), (20, 50), (20, 50), (20,50 ), (20,50 ), ],
                       yaw_range =(0,0),
                       ):

    goal_points = []
    center_p = np.array(root_position) + np.array([np.random.uniform(-rand_init,rand_init ),0, np.random.uniform(-rand_init,rand_init )])

    yaw = np.random.uniform(yaw_range[0], yaw_range[1])
    rot = transformations.rotation_matrix(yaw, [0,1,0])
    point = np.concatenate((np.array(norm_dir), np.array([1]))).reshape((-1,1))
    new_direction = rot.dot(point)
    new_direction = new_direction[0:3,0]

    num_goals = 5
    side = np.random.choice(2)
    for i in range( num_goals):
        final_p = generate_point_in_arc_area(center_p, new_direction,  theta= theta_list[i], dis_range= dis_list[i], side=side )

        x = final_p[0]
        z = final_p[2]

        goal_points.append( [x, 0+0.2, z])
        center_p[0] = x
        center_p[2] = z

        # norm_dir = final_p - center_p
        # norm_dir = norm_dir/np.linalg.norm(norm_dir)

    goal_points = np.array(goal_points)

    f2 = interp1d(goal_points[:,0], goal_points[:,2], kind='linear')
    x_sampled = np.linspace(goal_points[0,0], goal_points[-1,0], num=T, endpoint=True)
    z_sampled = f2(x_sampled)

    goal_traj_pos = np.concatenate((x_sampled, np.zeros_like(x_sampled), z_sampled)).reshape((3, -1)).transpose()

    return goal_traj_pos.flatten()


def sample_any_circle(
        root_position,
        norm_dir,
        T = 2 ,#second
        num_N = 24,
        vel_range = (100, 250),
        R_range = (100, 1000),
        direction = np.random.choice(2),
        theta = None,
        output_ignore_y = False,
        rand_init = 50,
):
    assert np.array(norm_dir).shape[0] == 3

    R = np.random.uniform(R_range[0], R_range[1])
    vel = np.random.uniform(vel_range[0], vel_range[1])

    if theta is None:
        theta = vel*T/R
    else:
        T = R*theta/vel
        num_N = int(12*T)

    if direction == 1:
        t = np.linspace(-np.pi/2, -np.pi/2+theta, num=num_N, endpoint=True)
        x = 0 + R*np.cos(t)
        y = R + R*np.sin(t)
    else:
        t = np.linspace( np.pi/2-theta, np.pi/2, num=num_N, endpoint=True)
        x = 0 + R*np.cos(t)
        y = -R + R*np.sin(t)

    goal_positions = np.concatenate((x, np.zeros_like(x), y)).reshape((3,-1)).transpose()

    # ---- general
    rot = transformations.rotation_matrix(np.arctan2(norm_dir[2], norm_dir[0]), [0,1,0])[0:3,0:3]
    goal_positions_transfered = np.linalg.inv(rot).dot(goal_positions.transpose()).transpose()
    goal_positions_transfered += (np.array(root_position)  + np.array([np.random.uniform(-rand_init,rand_init ),0, np.random.uniform(-rand_init,rand_init )]) )


    if output_ignore_y:
        goal_positions_transfered = goal_positions_transfered[:,[0,2]]
    return goal_positions_transfered

def sample_any_straigtLine(
        root_position,
        norm_dir,
        T = 2 ,#second
        num_N = 24,
        vel_range = (100, 250),
        theta = None,
        length = None,
        output_ignore_y = False,
        rand_init = 0,
):
    assert np.array(norm_dir).shape[0] == 3

    if theta is  None:
        theta =  np.random.uniform(0, 2*np.pi)
    vel  = np.random.uniform(vel_range[0], vel_range[1])

    if length is None:
        length = vel*T
    else:
        T = length/vel
    num_N = int(12*T)

    t = np.linspace(0, T, num=num_N, endpoint=True)
    x =  np.cos(theta)* t * vel
    y =  np.sin(theta) * t * vel

    goal_positions = np.concatenate((x, np.zeros_like(x), y)).reshape((3,-1)).transpose()

    # ---- general
    rot = transformations.rotation_matrix(np.arctan2(norm_dir[2], norm_dir[0]), [0,1,0])[0:3,0:3]
    goal_positions_transfered = np.linalg.inv(rot).dot(goal_positions.transpose()).transpose()
    goal_positions_transfered += (np.array(root_position)  + np.array([np.random.uniform(-rand_init,rand_init ),0, np.random.uniform(-rand_init,rand_init )]) )


    if output_ignore_y:
        goal_positions_transfered = goal_positions_transfered[:,[0,2]]
    return goal_positions_transfered


import matplotlib.pyplot as plt
def velocity_command_generator(max_step,dt,  hold_time, delta_list=[0.025,0.025, 0.025] ,
                      vx_range = (-0.8, 0.8), vy_range = (-0.8, 0.8), wyaw_range = (-0.8, 0.8),
                      render = False, seed = None):
    if seed is not None:
        np.random.seed(seed)
    vx_range = vx_range
    vy_range = vy_range
    wyaw_range = wyaw_range

    command_vx = []
    command_vy = []
    command_wyaw = []

    num_points = int(max_step/int(hold_time/dt))

    # vx_p_range = int(vx_range[1]*10)
    # vy_p_range = int(vy_range[1] * 10)
    # wyaw_p_range = int(wyaw_range[1] * 10)

    vx_p_range = int((vx_range[1] - vx_range[0])/delta_list[0])
    vy_p_range = int((vy_range[1] - vy_range[0]) / delta_list[1])
    wyaw_p_range = int((wyaw_range[1] - wyaw_range[0]) / delta_list[2])

    # vx_p = np.random.uniform(vx_range[0], vx_range[1], num_points)
    # vy_p = np.random.uniform(vy_range[0], vy_range[1], num_points)
    # wyaw_p = np.random.uniform(wyaw_range[0], wyaw_range[1], num_points)

    vx_p = np.random.randint(0, vx_p_range + 1, num_points) * delta_list[0] + vx_range[0]
    vy_p = np.random.randint(0, vy_p_range + 1, num_points) * delta_list[1] + vy_range[0]
    wyaw_p =np.random.randint(0, wyaw_p_range + 1, num_points) * delta_list[2] + wyaw_range[0]

    step = 0
    p_index =0
    time_done =0
    while step < max_step:

        if time_done < hold_time:
            command_vx.append(vx_p[p_index])
            command_vy.append(vy_p[p_index])
            command_wyaw.append(wyaw_p[p_index])
        else:
            time_done =0
            p_index += 1

        time_done += dt
        step +=1



    command_vx = np.array(command_vx)
    command_vy = np.array(command_vy)
    command_wyaw = np.array(command_wyaw)



    command = np.array([command_vx, command_vy, command_wyaw]).T
    if render:
        ax1 = plt.subplot(311)
        plt.plot(command_vx, color='red', label='o_1')
        plt.grid()
        plt.title('Vx')


        ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
        plt.plot(command_vy, color='red', label='o_1')
        plt.grid()
        plt.title('Vy')

        ax2 = plt.subplot(313, sharex=ax1, sharey=ax1)
        plt.plot(command_wyaw, color='red', label='o_1')
        plt.grid()
        plt.title('Wyaw')


        plt.show()


    return  command
