from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, glob
import re
import numpy as np
from tqdm import tqdm
import torch
import smplx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
import json
import csv
import pdb
import pickle
from grab_marker_seperate import get_grab_marker
import pandas as pd
import trimesh
import ast
import pyrender
print(os.getcwd())
from utils.PerspectiveCamera_holo import PerspectiveCamera_holo
# import osp
grab = 1
device = 'cuda:0'
def get_model2data(model_type):
    return smpl_to_openpose(model_type=model_type, use_hands=True, use_face=True, use_face_contour=False, openpose_format='coco25')
def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            # ex: body_mapping[0]=55: smplx joint 55 = openpose joint 0
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)   # len of 25
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)  # 21 joints for each hand
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)   # len of 51
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))
def get_body_model(type, gender, batch_size,device='cpu',joint_mapper = None):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model_path = '/home/yuxinyao/body_models/'
    body_model = smplx.create(body_model_path, model_type=type,
                                    joint_mapper=joint_mapper,
                                    gender=gender, ext='pkl',
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

class JointMapper(torch.nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)

def get_new_coordinate(body_mesh_model, betas, transl, pose,global_orient):
    '''
    this function produces transform from body local coordinate to the world coordinate.
    it takes only a single frame.
    local coodinate:
        - located at the pelvis
        - x axis: from left hip to the right hip
        - z axis: point up (negative gravity direction)
        - y axis: pointing forward, following right-hand rule
    '''
    temp_bodyconfig = {}
    temp_bodyconfig['transl'] = torch.FloatTensor(transl).cuda()
    temp_bodyconfig['global_orient'] = torch.FloatTensor(global_orient).cuda()
    temp_bodyconfig['body_pose'] = torch.FloatTensor(pose).cuda()
    temp_bodyconfig['betas'] = torch.FloatTensor(betas).cuda()
    smplxout = body_mesh_model(**temp_bodyconfig)
    joints = smplxout.joints.squeeze().detach().cpu().numpy()
    x_axis = joints[2,:] - joints[1,:]
    x_axis[-1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0,0,1])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis/np.linalg.norm(y_axis)
    global_ori_new = np.stack([x_axis, y_axis, z_axis], axis=1)
    transl_new = joints[:1,:] # put the local origin to pelvis

    return global_ori_new, transl_new

def get_new_coordinate_rotate(bm_one, transl,global_orient,pose,betas):
    temp_bodyconfig = {}
    temp_bodyconfig['transl'] = torch.FloatTensor(transl).cuda()
    temp_bodyconfig['global_orient'] = torch.FloatTensor(global_orient).cuda()
    temp_bodyconfig['body_pose'] = torch.FloatTensor(pose).cuda()
    temp_bodyconfig['betas'] = torch.FloatTensor(betas).cuda()

    smplxout = bm_one(**temp_bodyconfig)
    joints = smplxout.joints.squeeze().detach().cpu().numpy()

    global_ori_rotate = R.from_rotvec([np.pi/2,0,0]).as_matrix()
    transl_new = joints[:1,:] # put the local origin to pelvis

    return global_ori_rotate,transl_new


def calc_calibrate_offset(body_mesh_model, betas, transl, pose,global_orient):
    '''
    The factors to influence this offset is not clear. Maybe it is shape and pose dependent.
    Therefore, we calculate such delta_T for each individual body mesh.
    It takes a batch of body parameters
    input:
        body_params: dict, basically the input to the smplx model
        smplx_model: the model to generate smplx mesh, given body_params
    Output:
        the offset for params transform
    '''
    n_batches = transl.shape[0]
    temp_bodyconfig_seq = {}
    temp_bodyconfig_seq['transl'] = torch.zeros([n_batches,3], dtype=torch.float32).cuda()
    temp_bodyconfig_seq['global_orient'] = torch.zeros([n_batches,3], dtype=torch.float32).cuda()
    temp_bodyconfig_seq['body_pose'] = torch.FloatTensor(pose).cuda()
    temp_bodyconfig_seq['betas'] = torch.FloatTensor(betas).cuda()
    smplx_out = body_mesh_model(return_verts=True, **temp_bodyconfig_seq)
    delta_T = smplx_out.joints[:,0,:] # we output all pelvis locations
    delta_T = delta_T.detach().cpu().numpy() #[t, 3]

    return delta_T


def create_camera(camera_type='persp', **kwargs):
    # if camera_type.lower() == 'persp':
    #     return PerspectiveCamera(**kwargs)
    if camera_type.lower() == 'persp_holo':
        return PerspectiveCamera_holo(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(camera_type))


egobody_dataset_path = '/home/yuxinyao/datasets/egobody/smplx_camera_wearer'
with open('/home/yuxinyao/body_models/Mosh_related/CMU.json') as f:
        marker_cmu_41 = list(json.load(f)['markersets'][0]['indices'].values())
with open('/home/yuxinyao/body_models/Mosh_related/SSM2.json') as f:
        marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())

with open('/home/yuxinyao/body_models/Mosh_related/smplx_markerset_new.json') as f:
        marker_grab_67 = json.load(f)['markersets']
if grab:
    marker_ssm_67 = marker_ssm_67 + get_grab_marker(['finger','hand','palm_22'])
    marker_ssm_67_grab = marker_ssm_67 + get_grab_marker(['finger','hand','palm_22'])

# folder_seqs = glob.glob(os.path.join(egobody_dataset_path,"recording*"))
folder_seqs = glob.glob(os.path.join('/home/yuxinyao/datasets/egobody/smplx_interactee',"recording*"))
folder_dir = os.path.join('/home/yuxinyao/datasets/egobody/smplx_interactee')
egocentric_color_seqs = glob.glob(os.path.join(egobody_dataset_path[0:-19],"egocentric_color","recording*"))
egocentric_dir = os.path.join(egobody_dataset_path[0:-19],"egocentric_color")
calibration_seqs = glob.glob(os.path.join(egobody_dataset_path[0:-19],"calibration","recording*"))
calib_trans_dir = os.path.join(egobody_dataset_path[0:-19],"calibrations")
release_data_root = "/home/yuxinyao/datasets/egobody"
df = pd.read_csv(os.path.join(release_data_root, 'data_info_release.csv'))
recording_name_list = list(df['recording_name'])
start_frame_list = list(df['start_frame'])
end_frame_list = list(df['end_frame'])
body_idx_fpv_list = list(df['body_idx_fpv'])

body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))
start_frame_dict = dict(zip(recording_name_list, start_frame_list))
end_frame_dict = dict(zip(recording_name_list, end_frame_list))


N_MPS = 1
len_subseq = 10*N_MPS
bm_one_male = get_body_model('smplx','male',1,device = 'cuda')
bm_one_female = get_body_model('smplx','female',1,device='cuda')
bm_batch_male = get_body_model('smplx','male',len_subseq,device='cuda')
bm_batch_female = get_body_model('smplx','female',len_subseq,device='cuda')

# vertices = []
# bm = []
for index, record in enumerate(tqdm(folder_seqs)):
    # output = {}
    record_name = record.split("/")[-1]
    seqs = sorted(glob.glob(os.path.join(record, '*/*/*/*.pkl')))
    fpv_recording_dir = os.path.join(egocentric_dir,record_name)
    interactee_gender = body_idx_fpv_dict[record_name]
    holo2kinect_dir = os.path.join(calib_trans_dir, record_name,'cal_trans','holo_to_kinect12.json')
    with open(holo2kinect_dir,'r') as f:
        trans_holo2kinect = np.array(json.load(f)['trans'])
    trans_kinect2holo = np.linalg.inv(trans_holo2kinect)
    # output['trans_kinect2holo'] = trans_holo2kinect
    valid_frame_npz = glob.glob(os.path.join(egocentric_dir,record_name, '*/valid_frame.npz'))[0]
    kp_npz = glob.glob(os.path.join(egocentric_dir,record_name, '*/keypoints.npz'))[0]

    holo_pv_path_list = glob.glob(os.path.join(egocentric_dir,record_name, '*/PV', '*_frame_*.jpg'))
    # print(holo_pv_path_list)
    holo_pv_path_list = sorted(holo_pv_path_list)
    holo_frame_id_list = [x.split('/')[-1].split('_',1)[1][0:-4] for x in holo_pv_path_list]
    holo_frame_id_dict = dict(zip(holo_frame_id_list, holo_pv_path_list))

    valid_frames = np.load(valid_frame_npz)
    holo_2djoints_info = np.load(kp_npz)
    # print([x.split('/')[-1] for x in holo_2djoints_info['imgname']])
    holo_frame_id_all = [x.split('/')[-1].split('_',1)[1][0:-4] for x in valid_frames['imgname']]
    holo_valid_dict = dict(zip(holo_frame_id_all, valid_frames['valid']))  # 'frame_01888': True
    holo_frame_id_valid = [x.split('/')[-1].split('_', 1)[1][0:-4] for x in holo_2djoints_info['imgname']]  # list of all valid frame names (e.x., 'frame_01888')
    holo_keypoint_dict = dict(zip(holo_frame_id_valid, holo_2djoints_info['keypoints']))
    # print("holo_pv_path_list[0]" + holo_pv_path_list[0])
    holo_timestamp_list = [x.split('/')[-1].split('_')[0]for x in holo_pv_path_list]
    # print("holo_timesstamp_list[0]" + holo_timestamp_list[0])
    holo_timestamp_dict = dict(zip(holo_timestamp_list, holo_frame_id_list))
    pv_info_path = glob.glob(os.path.join(fpv_recording_dir, '*/*_pv.txt'))[0]
    with open(pv_info_path) as f:
        lines = f.readlines()
    holo_cx, holo_cy, holo_w, holo_h = ast.literal_eval(lines[0])  # hololens pv camera infomation

    holo_fx_dict = {}
    holo_fy_dict = {}
    holo_pv2world_trans_dict = {}
    for i, frame in enumerate(lines[1:]):
        frame = frame.split((','))
        cur_timestamp = frame[0]  # string
        cur_fx = float(frame[1])
        cur_fy = float(frame[2])
        cur_pv2world_transform = np.array(frame[3:20]).astype(float).reshape((4, 4))

        if cur_timestamp in holo_timestamp_dict.keys():
            cur_frame_id = holo_timestamp_dict[cur_timestamp]
            holo_fx_dict[cur_frame_id] = cur_fx
            holo_fy_dict[cur_frame_id] = cur_fy
            holo_pv2world_trans_dict[cur_frame_id] = cur_pv2world_transform


    joint_mapper = JointMapper(get_model2data(model_type='smplx'))
    interactee_gender = body_idx_fpv_dict[record_name].split(' ')[1]

    transl_seq = np.zeros((len_subseq,3))
    pose_seq = np.zeros((len_subseq,63))
    betas_seq = np.zeros((len_subseq,10))
    left_hand_seq = np.zeros((len_subseq,12))
    right_hand_seq = np.zeros((len_subseq,12))
    jaw_pose_seq = np.zeros((len_subseq,3))
    leye_pose_seq = np.zeros((len_subseq,3))
    reye_pose_seq = np.zeros((len_subseq,3))
    expression_seq = np.zeros((len_subseq,10))
    betas = np.zeros((len_subseq,10))
    global_orient_seq = np.zeros((len_subseq,3))
    kp_seq = np.zeros((len_subseq,25,3))
    cur_fx_seq = np.zeros((len_subseq,1))
    cur_fy_seq = np.zeros((len_subseq,1))
    holo_cx_seq = np.zeros((len_subseq,1))
    holo_cy_seq = np.zeros((len_subseq,1))
    cur_world2pv_transform_seq= np.zeros((len_subseq,4,4))
    camera_holo_kp_seq = []
    address_seq = [0]*len_subseq


    # calib_seq = 

    bodymodel_batch = bm_batch_male if interactee_gender == 'male' else bm_batch_female
    bodymodel_one = bm_one_male if interactee_gender =='male' else bm_one_female
    
    counter = 0
    next = False
    #for each frame 
    previous_frame = start_frame_dict[record_name]

    for i_frame in range(start_frame_dict[record_name], end_frame_dict[record_name],1):
        # print("i_frame: {}".format(i_frame))
        holo_frame_id = 'frame_{}'.format("%05d" % i_frame)

        if (end_frame_dict[record_name] - i_frame<10 ):
            break
        # get 2d keypoints, must have 2d points&calib&body param
        if (holo_frame_id in holo_valid_dict.keys()) and holo_valid_dict[holo_frame_id]:
            kp = holo_keypoint_dict[holo_frame_id]
            # print("kp: {}".format(kp.shape))
            # print("kp: {}".format(kp))

            datadir = glob.glob(os.path.join(folder_dir,record_name,'*/results',holo_frame_id,'*.pkl'))
            if datadir ==[]:
                # print(os.path.join(folder_dir,record_name,'*/results',holo_frame_id,'*.pkl'),"not exist")
                continue

            with open(datadir[0],'rb') as f:
                # print('open file: {}'.format(datadir[0]))
                data = pickle.load(f)
            if i_frame -previous_frame>3:
                next = True

            previous_frame = i_frame
            if holo_frame_id not in holo_frame_id_dict.keys():  # the frame is dropped in hololens recording
                # print('pass')
                continue
            # else:
            fpv_img_path = os.path.join(egocentric_dir,record_name, 'PV', holo_frame_id_dict[holo_frame_id])
            # print(fpv_img_path)
            cur_fx = holo_fx_dict[holo_frame_id]
            cur_fy = holo_fy_dict[holo_frame_id]
            cur_pv2world_transform = holo_pv2world_trans_dict[holo_frame_id]
            cur_world2pv_transform = np.linalg.inv(cur_pv2world_transform)
            
            


            
            # cur_world2pv_transform_seq.append(cur_world2pv_transform)
            # camera_holo_kp_seq.append(create_camera(camera_type='persp_holo',
                                                #    focal_length_x=torch.tensor([cur_fx]).to(device).unsqueeze(0),
                                                #    focal_length_y=torch.tensor([cur_fy]).to(device).unsqueeze(0),
                                                #    center= torch.tensor([holo_cx, holo_cy]).view(-1,2),
                                                #    batch_size=1).to(device=device))
            # print("cur_fx.shape: {}".format(cur_fx.shape))
            # print("holo_cx.shape: {}".format(holo_cx.shape))


            
            transl = data['transl']
            pose = data['body_pose']
            global_orient = data['global_orient']
            betas = data['betas'][:10]
            # print(counter)
            transl_seq[counter,:] = transl
            pose_seq[counter,:]= pose
            global_orient_seq[counter,:] = global_orient
            betas_seq[counter,:] = betas
            # print("cur_world2pv_transform: {}".format(cur_world2pv_transform.shape))
            # print("cur_world2pv_transform_seq.shape: {}".format(cur_world2pv_transform_seq.shape))
            cur_world2pv_transform_seq[counter,:,:] = cur_world2pv_transform
            left_hand_seq[counter,:] = data['left_hand_pose']
            right_hand_seq[counter,:] = data['right_hand_pose']
            jaw_pose_seq[counter,:] = data['jaw_pose']
            leye_pose_seq[counter,:] = data['leye_pose']
            reye_pose_seq[counter,:] = data['reye_pose']
            expression_seq[counter,:] = data['expression']
            kp_seq[counter,:,:] = kp
            
            cur_fx_seq[counter] = cur_fx
            cur_fy_seq[counter] = cur_fy
            holo_cx_seq[counter] = holo_cx
            holo_cy_seq[counter] = holo_cy

            address_seq[counter] = fpv_img_path

            # camera_holo_kp_seq

            if next == True:
                #reinitialize the series
                transl_seq = np.zeros((len_subseq,3))
                pose_seq = np.zeros((len_subseq,63))
                betas_seq = np.zeros((len_subseq,10))
                left_hand_seq = np.zeros((len_subseq,12))
                right_hand_seq = np.zeros((len_subseq,12))
                jaw_pose_seq = np.zeros((len_subseq,3))
                leye_pose_seq = np.zeros((len_subseq,3))
                reye_pose_seq = np.zeros((len_subseq,3))
                expression_seq = np.zeros((len_subseq,10))
                betas = np.zeros((len_subseq,10))
                global_orient_seq = np.zeros((len_subseq,3))
                kp_seq = np.zeros((len_subseq,25,3))
                trans_kinect2holo_seq = []
                cur_world2pv_transform_seq= np.zeros((len_subseq,4,4))
                camera_holo_kp_seq = []
                cur_fx_seq = np.zeros((len_subseq,1))
                cur_fy_seq = np.zeros((len_subseq,1))
                holo_cx_seq = np.zeros((len_subseq,1))
                holo_cy_seq = np.zeros((len_subseq,1))
                counter = 0
                address = []
                next = False
                # print("jump frame:",holo_frame_id)
                continue 

            if counter == 9:

                data_out = {}

                outfolder = "/home/yuxinyao/datasets/egobody/canicalized-camera-wearer-grab-openpose_septest/" + record_name
                if not os.path.exists(outfolder):
                    os.makedirs(outfolder)
                outfilename = os.path.join(outfolder, 'subseq_{:05d}.pkl'.format(i_frame))

                transf_rotate_xaxis, transf_transl_xaxis = get_new_coordinate_rotate(bodymodel_one,transl_seq[:1],global_orient_seq[:1],pose_seq[:1],betas_seq[:1])

                delta_T_rotate = calc_calibrate_offset(bodymodel_batch, betas_seq, transl_seq, pose_seq,global_orient_seq)

                global_ori_rotate = R.from_matrix(np.einsum('ij,tjk->tik', transf_rotate_xaxis.T, R.from_rotvec(global_orient_seq).as_matrix())).as_rotvec()
                transl_rotate = np.einsum('ij,tj->ti', transf_rotate_xaxis.T, transl_seq+delta_T_rotate-transf_transl_xaxis)-delta_T_rotate

                #calibrate
                transf_rotmat, transf_transl = get_new_coordinate(bodymodel_one, betas_seq[:1,:], transl_rotate[:1,:], pose_seq[:1,:],global_ori_rotate[:1,:])

                delta_T = calc_calibrate_offset(bodymodel_batch, betas_seq, transl_rotate, pose_seq,global_ori_rotate)
                
                global_ori_save = R.from_matrix(np.einsum('ij,tjk->tik', transf_rotmat.T, R.from_rotvec(global_ori_rotate).as_matrix())).as_rotvec()
                transl_save = np.einsum('ij,tj->ti', transf_rotmat.T, transl_rotate+delta_T-transf_transl)-delta_T
                
                print("translsave: {}".format(transl_save))
                print("global_ori_save: {}".format(global_ori_save))
                data_out['address'] = address_seq
                data_out['trans'] = transl_save
                data_out['body_pose'] = pose_seq
                data_out['betas'] = betas_seq
                data_out['gender'] = data['gender']
                data_out['global_orient'] = global_ori_save
                data_out['mocap_framerate'] = 30
                data_out['transf_rotmat'] = transf_rotmat
                data_out['transf_transl'] = transf_transl
                data_out['delta_T'] = delta_T
                data_out['transf_rotate_xaxis'] = transf_rotate_xaxis
                data_out['transf_transl_xaxis'] = transf_transl_xaxis
                data_out['delta_T_rotate'] = delta_T_rotate
                data_out['left_hand_pose'] = left_hand_seq
                data_out['right_hand_pose'] = right_hand_seq
                data_out['jaw_pose'] = jaw_pose_seq
                data_out['leye_pose'] = leye_pose_seq
                data_out['reye_pose'] = reye_pose_seq
                data_out['poses'] = np.concatenate([global_ori_save, pose_seq],axis = 1)
                data_out['kp']= kp_seq
                data_out['cur_world2pv_transform'] = cur_world2pv_transform_seq
                # data_out['camera_holo_kp']= camera_holo_kp_seq
                # print(len(data_out['camera_holo_kp']))
                data_out['trans_kinect2holo'] = trans_kinect2holo
                # print('kinect cur_world2pv_transform holo:{}'.format(data_out['cur_world2pv_transform']))
                data_out['cur_fx'] = cur_fx_seq
                data_out['cur_fy'] = cur_fy_seq
                data_out['holo_cx'] = holo_cx_seq
                data_out['holo_cy'] = holo_cy_seq
                # print("address len:"+str(len(data_out['address'])))
                # print('dataout[\'curfx\'].shape:{}'.format(data_out['cur_fx']))
                # print('dataout[\'holo_cx\'].shape:',data_out['holo_cx'].shape)

                body_param = {}
                body_param['transl'] = torch.FloatTensor(transl_save).cuda()
                body_param['global_orient'] = torch.FloatTensor(global_ori_save).cuda()
                body_param['betas'] = torch.FloatTensor(betas_seq).cuda()
                body_param['body_pose'] = torch.FloatTensor(pose_seq).cuda()
                body_param['left_hand_pose'] = torch.FloatTensor(left_hand_seq).cuda()
                body_param['right_hand_pose'] = torch.FloatTensor(right_hand_seq).cuda()
                body_param['jaw_pose'] = torch.FloatTensor(jaw_pose_seq).cuda()
                body_param['leye_pose'] = torch.FloatTensor(leye_pose_seq).cuda()
                body_param['reye_pose'] = torch.FloatTensor(reye_pose_seq).cuda()
                body_param['expression'] = torch.FloatTensor(expression_seq).cuda()

                smplxout = bodymodel_batch(return_verts=True, **body_param)
                vertices = smplxout.vertices.detach().cpu().numpy()
                joints = smplxout.joints[:,:22,:].detach().squeeze().cpu().numpy()
                markers_67 = smplxout.vertices[:,marker_ssm_67,:].detach().squeeze().cpu().numpy()
                markers_41 = smplxout.vertices[:,marker_cmu_41,:].detach().squeeze().cpu().numpy()
                marker_grab = smplxout.vertices[:,marker_ssm_67_grab,:].detach().squeeze().cpu().numpy()
                data_out['joints'] = joints
                
                data_out['marker_ssm2_67'] = markers_67
                data_out['marker_ssm2_67_grab'] = marker_grab
                data_out['marker_cmu_41'] = markers_41
                with open(outfilename, 'wb') as f:
                    pickle.dump(data_out,f)
                # print(outfilename)
                transl_seq = np.zeros((len_subseq,3))
                pose_seq = np.zeros((len_subseq,63))
                betas_seq = np.zeros((len_subseq,10))
                left_hand_seq = np.zeros((len_subseq,12))
                right_hand_seq = np.zeros((len_subseq,12))
                jaw_pose_seq = np.zeros((len_subseq,3))
                leye_pose_seq = np.zeros((len_subseq,3))
                reye_pose_seq = np.zeros((len_subseq,3))
                expression_seq = np.zeros((len_subseq,10))
                betas = np.zeros((len_subseq,10))
                global_orient_seq = np.zeros((len_subseq,3))
                kp_seq = np.zeros((len_subseq,25,3))
                trans_kinect2holo_seq = []
                cur_world2pv_transform_seq= np.zeros((len_subseq,4,4))
                camera_holo_kp_seq = []
                address_seq = [0]*10
                cur_fx_seq = np.zeros((len_subseq,1))
                cur_fy_seq = np.zeros((len_subseq,1))
                holo_cx_seq = np.zeros((len_subseq,1))
                holo_cy_seq = np.zeros((len_subseq,1))

                counter = 0

                # print("outfilename:",outfilename)
            else: counter +=1



    # break
