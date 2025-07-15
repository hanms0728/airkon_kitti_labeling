import numpy as np
import cv2
import math
import yaml
from pyquaternion import Quaternion

def read_kitti_cal(calfile):
    text_file = open(calfile, 'r')
    for line in text_file:
        parsed = line.strip().split()
        if parsed[0] == 'P2:':
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, :] = [float(x) for x in parsed[1:5]]
            p2[1, :] = [float(x) for x in parsed[5:9]]
            p2[2, :] = [float(x) for x in parsed[9:13]]
            p2[3, 3] = 1
    text_file.close()
    return p2

def load_denorm_data(denormfile):
    with open(denormfile, 'r') as f:
        line = f.readline()
        a, b, c = map(float, line.strip().split()[:3])
    return np.array([a,b,c])

def compute_c2g_trans(de_norm):
    ground_z_axis = de_norm 
    cam_xaxis = np.array([1.0, 0.0, 0.0])
    ground_x_axis = cam_xaxis - cam_xaxis.dot(ground_z_axis) * ground_z_axis
    ground_x_axis /= np.linalg.norm(ground_x_axis)
    ground_y_axis = np.cross(ground_z_axis, ground_x_axis)
    ground_y_axis /= np.linalg.norm(ground_y_axis)
    return np.vstack([ground_x_axis, ground_y_axis, ground_z_axis])

def read_kitti_ext(extfile):
    with open(extfile, 'r') as f:
        x = yaml.safe_load(f)
    r = x['transform']['rotation']
    t = x['transform']['translation']
    q = Quaternion(r['w'], r['x'], r['y'], r['z'])
    m = q.rotation_matrix
    m = np.matrix(m).reshape((3, 3))
    t = np.matrix([t['x'], t['y'], t['z']]).T
    p1 = np.vstack((np.hstack((m, t)), np.array([0, 0, 0, 1])))
    return np.array(p1.I)

from show_2d3d_box import project_3d_ground, project_3d_world

def load_scene_context(base_dir, name):
    p2 = read_kitti_cal(f"{base_dir}/calib/{name}.txt")
    extrinsics = f"{base_dir}/extrinsics/{name}.yaml"
    denorm = load_denorm_data(f"{base_dir}/denorm/{name}.txt")
    return p2, extrinsics, denorm

def draw_label_on_image(img, p2, extrinsics, denorm, mode, h,w,l,X,Y,Z,ry):
    if mode == 'Ground':
        c2g_trans = compute_c2g_trans(denorm)
        verts3d = project_3d_ground(p2, np.array([X,Y,Z]), w,h,l,ry, denorm, c2g_trans)
    else:
        camera2world = np.linalg.inv(read_kitti_ext(extrinsics))
        bottom_center_in_world = camera2world @ np.array([[X],[Y],[Z],[1]])
        verts3d = project_3d_world(p2, bottom_center_in_world, w,h,l,ry,camera2world)

    if verts3d is None:
        return img
    verts3d = verts3d.astype(int)
    # 차량 정면 사각형을 파란색으로 표시
    for s, e in [(0,1),(1,5),(5,4),(4,0)]:
        cv2.line(img, tuple(verts3d[s]), tuple(verts3d[e]), (255,0,0), 2)
    # 나머지 라인은 빨간색
    for s, e in [(2,1),(0,3),(2,3),(7,4),(5,6),(6,7),(7,3),(2,6)]:
        cv2.line(img, tuple(verts3d[s]), tuple(verts3d[e]), (0,0,255), 2)
    return img