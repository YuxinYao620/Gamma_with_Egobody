from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, glob
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
grab = 1
def get_body_model(type, gender, batch_size,device='cpu'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model_path = '/home/yuxinyao/body_models/'
    body_model = smplx.create(body_model_path, model_type=type,
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

    smplxout = bodymodel_one(**temp_bodyconfig)
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




egobody_dataset_path = '/home/yuxinyao/datasets/egobody/smplx_camera_wearer'
with open('/home/yuxinyao/body_models/Mosh_related/CMU.json') as f:
        marker_cmu_41 = list(json.load(f)['markersets'][0]['indices'].values())

with open('/home/yuxinyao/body_models/Mosh_related/SSM2.json') as f:
        marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())
if grab:
    marker_ssm_67 = marker_ssm_67 + get_grab_marker(['finger','hand','palm_22'])

folder_seqs = glob.glob(os.path.join(egobody_dataset_path,"recording*"))

len_subseq = 10
bm_one_male = get_body_model('smplx','male',1,device = 'cuda')
bm_one_female = get_body_model('smplx','female',1,device='cuda')
bm_batch_male = get_body_model('smplx','male',len_subseq,device='cuda')
bm_batch_female = get_body_model('smplx','female',len_subseq,device='cuda')

# vertices = []
# bm = []

for record in tqdm(folder_seqs):
    seqs = sorted(glob.glob(os.path.join(record, '*/*/*/*.pkl')))
    for i in range(0, len(seqs),len_subseq):
        # print("folder len"+ str(len(folder_seqs)))
        if i+len_subseq > len(seqs):
            continue # skip the last few frames
        first_frame_path = seqs[i]
        with open(first_frame_path,'rb') as f:
            first_frame = dict(pickle.load(f))
        transl_seq = np.zeros((len_subseq,3))
        pose_seq = np.zeros((len_subseq,63))
        betas_seq = np.zeros((len_subseq,10))
        betas = first_frame['betas']
        global_orient_seq = np.zeros((len_subseq,3))
        bodymodel_batch = bm_batch_male if str(first_frame['gender']) =='male' else bm_batch_female
        bodymodel_one = bm_one_male if str(first_frame['gender']) =='male' else bm_one_female

        for j in range(0,len_subseq):
            seq = seqs[i+j]
            # print("i, j:{},{}".format(i,j))
            # print("seqs[{}]:{} ".format(i+j, seqs[i+j]) +"\n")
            with open(seq,'rb') as f:
                data = dict(pickle.load(f))
            transl = data['transl']
            pose = data['body_pose']
            global_orient = data['global_orient']
            betas = data['betas'][:10]
            
            transl_seq[j,:] = transl
            pose_seq[j,:]= pose
            global_orient_seq[j,:] = global_orient
            betas_seq[j,:] = betas

        data_out = {}
        if grab:
            outfolder = "/home/yuxinyao/datasets/egobody/canicalized-camera-wearer-grab/" + str(record[52:])
        else:
            outfolder = "/home/yuxinyao/datasets/egobody/canicalized-camera-wearer/" + str(record[52:])
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        outfilename = os.path.join(outfolder, 'subseq_{:05d}.pkl'.format(i))

        transf_rotate_xaxis, transf_transl_xaxis = get_new_coordinate_rotate(bodymodel_one,first_frame['transl'],first_frame['global_orient'],first_frame['body_pose'],first_frame['betas'])

        delta_T_rotate = calc_calibrate_offset(bodymodel_batch, betas_seq, transl_seq, pose_seq,global_orient_seq)

        global_ori_rotate = R.from_matrix(np.einsum('ij,tjk->tik', transf_rotate_xaxis.T, R.from_rotvec(global_orient_seq).as_matrix())).as_rotvec()
        transl_rotate = np.einsum('ij,tj->ti', transf_rotate_xaxis.T, transl_seq+delta_T_rotate-transf_transl_xaxis)-delta_T_rotate

        #calibrate
        transf_rotmat, transf_transl = get_new_coordinate(bodymodel_one, betas_seq[:1,:], transl_rotate[:1,:], pose_seq[:1,:],global_ori_rotate[:1,:])
                ### calibrate offset
        delta_T = calc_calibrate_offset(bodymodel_batch, betas_seq, transl_rotate, pose_seq,global_ori_rotate)
        
        global_ori_save = R.from_matrix(np.einsum('ij,tjk->tik', transf_rotmat.T, R.from_rotvec(global_ori_rotate).as_matrix())).as_rotvec()
        transl_save = np.einsum('ij,tj->ti', transf_rotmat.T, transl_rotate+delta_T-transf_transl)-delta_T

        
        data_out['trans'] = transl_save
        data_out['body_pose'] = pose_seq
        data_out['betas'] = betas_seq
        data_out['gender'] = data['gender']
        data_out['global_orient'] = global_ori_save
        data_out['mocap_framerate'] = 30
        data_out['transf_rotmat'] = transf_rotmat
        data_out['transf_transl'] = transf_transl
        # print("global_ori_save.shape: {}, pose_seq.shape:{}".format(global_ori_save.shape, pose_seq.shape))
        data_out['poses'] = np.concatenate([global_ori_save, pose_seq],axis = 1)
        # print(data_out['poses'].shape)

        # break

        body_param = {}
        body_param['transl'] = torch.FloatTensor(transl_save).cuda()
        body_param['global_orient'] = torch.FloatTensor(global_ori_save).cuda()
        body_param['betas'] = torch.FloatTensor(betas_seq).cuda()
        body_param['body_pose'] = torch.FloatTensor(pose_seq).cuda()

# ### save body params :
#         body_param_save = {}
#         for k in body_param.keys():
#             body_param_save[k] = body_param[k].detach().cpu().numpy().tolist()
#         import json
#         print(os.getcwd())
#         with open('./body_param_canonicalization.txt','w') as f:
#             f.write(json.dumps(body_param_save))



        smplxout = bodymodel_batch(return_verts=True, **body_param)
        vertices = smplxout.vertices.detach().cpu().numpy()
        joints = smplxout.joints[:,:22,:].detach().squeeze().cpu().numpy()
        # print("transl:"+str(data_out['trans']))
        # print("global_orient: "+str(data_out['global_orient']))
        # print("first frame pelvis:"+str(joints[0,0,:]))
        markers_41 = smplxout.vertices[:,marker_cmu_41,:].detach().squeeze().cpu().numpy()
        markers_67 = smplxout.vertices[:,marker_ssm_67,:].detach().squeeze().cpu().numpy()
        # print(markers_67.shape)
        data_out['joints'] = joints
        data_out['marker_cmu_41'] = markers_41
        data_out['marker_ssm2_67'] = markers_67
        with open(outfilename, 'wb') as f:
            pickle.dump(data_out,f)
        # print(outfilename)

    #     break
    break

# with open("/home/yuxinyao/datasets/egobody/canicalized-camera-wearer/recording_20210910_S06_S05_01/subseq_00000.pkl",'rb') as f:
#     output_data = pickle.load(f)
# print("output_data[global_orient]: \n"+str(output_data['global_orient']))
# print("output_data[transl]:\n"+str(output_data['trans']))
# # print("outout_data['joints']"+str(output_data['joints'][0,0,:]))
# print("beta: "+str(data_out['betas']))

# import open3d as o3d
# import cv2
# vis = o3d.visualization.Visualizer()
# vis.create_window(width=960, height=540,visible=True)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(output_data['joints'].reshape(-1, 3))
# o3d.io.write_point_cloud("sync.ply", pcd)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(output_data['joints'].reshape(-1, 3))
# o3d.io.write_point_cloud("sync.ply", pcd)
# # vis.add_geometry(pcd)
# # vis.poll_events()
# # vis.update_renderer()

# body = o3d.geometry.TriangleMesh()
# # vis.add_geometry(body)
# # vis.poll_events()
# # vis.update_renderer()

# body.vertices = o3d.utility.Vector3dVector(vertices.reshape(-1,3))
# body.triangles = o3d.utility.Vector3iVector(bm_batch_male.faces)
# body.vertex_normals = o3d.utility.Vector3dVector([])
# body.triangle_normals = o3d.utility.Vector3dVector([])
# body.compute_vertex_normals()
# # vis.update_geometry(body)
# # vis.update_renderer()

# o3d.visualization.draw_geometries([body, pcd])

# rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
# cv2.imshow("frame2", np.uint8(255*rgb[:,:,[2,1,0]]))
# outfile_path = "/home/yuxinyao/datasets/egobody/canicalized-camera-wearer/recording_20210910_S06_S05_01/subseq_00000.png"
# frame_idx = 0
# if outfile_path is not None:
#     renderimgname = os.path.join(outfile_path, 'img_{:05d}.png'.format(frame_idx))
#     frame_idx = frame_idx + 1
#     cv2.imwrite(renderimgname, np.uint8(255*rgb[:,:,[2,1,0]]))
# cv2.waitKey(0)


# # pcd_load = o3d.io.read_point_cloud("sync.ply")
# # o3d.visualization.draw_geometries([pcd_load])
# # ply_point_cloud = o3d.data.PLYPointCloud()
# # pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
# # # print(pcd)
# # # print(np.asarray(pcd.points))
# # o3d.visualization.draw_geometries(pcd)
# # #                                   front=[0.4257, -0.2125, -0.8795],
# #                                   lookat=[2, 2, 2],
# #                                   up=[-0.0694, -0.9768, 0.2024])