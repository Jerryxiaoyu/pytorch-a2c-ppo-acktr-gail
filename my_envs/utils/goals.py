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
