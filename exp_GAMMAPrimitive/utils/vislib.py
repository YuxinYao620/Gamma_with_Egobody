import os
import sys
import numpy as np
import open3d as o3d
import torch
import smplx
from scipy.spatial.transform import Rotation as R
def update_render_cam(cam_param,  trans):
    ### NOTE: trans is the trans relative to the world coordinate!!!

    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T) #!!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0,0,0,1]])
    mat = np.concatenate([cam_R, cam_T],axis=-1)
    mat = np.concatenate([mat, cam_aux],axis=0)
    cam_param.extrinsic = mat
    return cam_param


def create_lineset(x_range, y_range, z_range):
    gp_lines = o3d.geometry.LineSet()
    gp_pcd = o3d.geometry.PointCloud()
    points = np.stack(np.meshgrid(x_range, y_range, z_range), axis=-1)

    lines = []
    for ii in range( x_range.shape[0]-1):
        for jj in range(y_range.shape[0]-1):
            lines.append(np.array([ii*x_range.shape[0]+jj, ii*x_range.shape[0]+jj+1]))
            lines.append(np.array([ii*x_range.shape[0]+jj, ii*x_range.shape[0]+jj+y_range.shape[0]]))

    points = np.reshape(points, [-1,3])
    colors = np.random.rand(len(lines), 3)*0.5+0.5

    gp_lines.points = o3d.utility.Vector3dVector(points)
    gp_lines.colors = o3d.utility.Vector3dVector(colors)
    gp_lines.lines = o3d.utility.Vector2iVector(np.stack(lines,axis=0))
    gp_pcd.points = o3d.utility.Vector3dVector(points)

    return gp_lines, gp_pcd


def color_hex2rgb(hex):
    h = hex.lstrip('#')
    return np.array(  tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) )/255


def get_body_model(type, gender, batch_size,device='cpu'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model_path = '/home/yuxinyao/body_models'
    body_model = smplx.create(body_model_path, model_type=type,
                                    gender=gender, ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=batch_size
                                    )
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model
def rotate_back(bparam,transf_rotate_xaxis,transf_transl_xaxis,delta_T_rotate,transf_rotmat,transf_transl,delta_T,to_numpy):
    if to_numpy == True:
        transl_save = bparam['transl'].detach().cpu().numpy()
        global_ori_save = bparam['global_orient'].detach().cpu().numpy()
    else:
        transl_save = bparam['transl']
        global_ori_save = bparam['global_orient']
    # print("transf_rotate_xaxis.shape:{}".format(transf_rotate_xaxis.shape))
    # print("transf_transl_xaxis.shape:{}".format(transf_transl_xaxis.shape))
    # print("delta_T_rotate.shape:{}".format(delta_T_rotate.shape))
    # print("transf_rotmat.shape:{}".format(transf_rotmat.shape))
    # print("transf_transl.shape:{}".format(transf_transl.shape))
    # print("delta_T.shape:{}".format(delta_T.shape))
    transl_rotate_recover  = ((np.linalg.inv(transf_rotmat.T))@((transl_save+delta_T).T)).T+transf_transl-delta_T
    global_ori_rotate_recover = R.from_matrix(np.einsum('ij,tjk->tik', np.linalg.inv(transf_rotmat.T),R.from_rotvec(global_ori_save).as_matrix())).as_rotvec()

    transl_seq_recover = ((np.linalg.inv(transf_rotate_xaxis.T))@((transl_rotate_recover+delta_T_rotate).T)).T +transf_transl_xaxis-delta_T_rotate
    global_orient_seq_recover = R.from_matrix(np.einsum('ij,tjk->tik', np.linalg.inv(transf_rotate_xaxis.T),R.from_rotvec(global_ori_rotate_recover).as_matrix())).as_rotvec()

    return transl_seq_recover,global_orient_seq_recover