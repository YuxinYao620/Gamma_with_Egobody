from importlib.metadata import requires
import torchgeometry as tgm
import time
from tkinter.messagebox import NO
import torch
import numpy as np
import os, sys, glob
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import pickle
import json
import random
import pdb
from tensorboardX import SummaryWriter
from typing import Union
from typing import Tuple
from PIL import ImageDraw
from models.baseops import MLP
from models.baseops import VAE
from models.baseops import TrainOP
from models.baseops import TestOP
from models.baseops import get_scheduler
from models.baseops import get_logger
# from models.baseops import get_body_model
from models.baseops import CanonicalCoordinateExtractor
from models.baseops import RotConverter
from exp_GAMMAPrimitive.utils.config_env import *
from models.models_GAMMA_primitive import *
from exp_GAMMAPrimitive.utils.PerspectiveCamera_holo import JointMapper
from exp_GAMMAPrimitive.utils.PerspectiveCamera_holo import PerspectiveCamera_holo,calc_calibrate_offset, get_new_coordinate_rotate
import matplotlib.pyplot as plts
from scipy.spatial.transform import Rotation as R
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
import PIL.Image
from PIL import ImageDraw
from torch.optim.lr_scheduler import MultiStepLR
import smplx
def draw(points,string):
    # pass
    A = PIL.Image.new('RGB', (2000, 2000), (255, 255, 255))
    draw = ImageDraw.Draw(A)
    # draw.point(points[0], fill=(255, 0, 0))
    # print(points[0])
    fill = (255,0,0) if 'kp' in string else (0,255,0)
    for seq in range(len(points)):
        if (points[0].shape[-1] == 2):
            for i, j in points[seq]:
                draw.ellipse((i-4,j-4,i+4,j+4), fill=fill)
                # print('i:{},j:{}'.format(i,j))
        else:
            for i, j,k in points[seq]:
                draw.ellipse((i-4,j-4,i+4,j+4), fill=fill)
                # print('i:{},j:{},k:{}'.format(i,j,k))
        A.save("/home/yuxinyao/2dimgs/2{}_{}_test.png".format(string,seq))




class GAMMAPrimitiveComboRecOP(TestOP):
    """the interface to GAMMA when using it to produce motions

    """
    def __init__(self, predictorcfg, regressorcfg, testconfig):
        self.dtype = torch.float32
        gpu_index = testconfig.get('gpu_index',0)
        self.device = torch.device('cuda',
                index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.predictorcfg = predictorcfg
        self.regressorcfg = regressorcfg
        self.testconfig = testconfig
        self.use_cont = self.regressorcfg.modelconfig.get('use_cont', False)
        self.t_his = self.predictorcfg.modelconfig['t_his']
        self.var_seed_len = self.predictorcfg.modelconfig.get('var_seed_len', False)

    def build_model(self, load_pretrained_model=False):
        # print("regressorcfg = {}".format(self.regressorcfg.modelconfig))
        self.model = GAMMAPrimitiveCombo(self.predictorcfg.modelconfig, self.regressorcfg.modelconfig)
        self.model.predictor.eval()
        self.model.regressor.eval()
        self.model.to(self.device)

        '''load pre-trained checkpoints'''
        if load_pretrained_model:
            ## for marker
            try:
                ckpt_path = os.path.join(self.predictorcfg.trainconfig['save_dir'],'epoch-300.ckp')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.model.predictor.load_state_dict(checkpoint['model_state_dict'])
                print('[INFO] --load pre-trained predictor: {}'.format(ckpt_path))
            except:
                ckpt_path = os.path.join(self.predictorcfg.trainconfig['save_dir'],'epoch-100.ckp')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.model.predictor.load_state_dict(checkpoint['model_state_dict'])
                print('[INFO] --load pre-trained predictor: {}'.format(ckpt_path))
            ## for regressor
            ckpt_path = os.path.join(self.regressorcfg.trainconfig['save_dir'],'epoch-100.ckp')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.regressor.load_state_dict(checkpoint['model_state_dict'])
            print('[INFO] --load pre-trained regressor: {}'.format(ckpt_path))
        else:
            try:
                ckpt_path = os.path.join(self.testconfig['ckpt_dir'],'epoch-120.ckp')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
            except:
                ckpt_path = os.path.join(self.testconfig['ckpt_dir'],'epoch-10.ckp')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print('[INFO] --load combo model: {}'.format(ckpt_path))

        z_dim = self.model.predictor.z_dim

        normal = Normal(torch.zeros(z_dim).cuda(), torch.ones(z_dim).cuda())
        self.zprior = Independent(normal, 1)  

    # def sample_prior(self,
    #                 X: torch.Tensor,
    #                 betas: torch.Tensor,
    #                 z: Union[None, torch.Tensor]=None,
    #     ) -> Tuple[torch.Tensor, torch.Tensor]: #[t, b, d]
    #     """generate motion primitives based on a motion seed X
    #     all inputs are in torch.FloatTensor on cuda
    #     all inputs are from the same gender.

    #     Args:
    #         - X: the motion seed, with the shape [t, b, d]
    #         - betas: [b,10] body shape
    #         - z: when None, we get random samples. Or we get the motions corresponding to z.

    #     Returns:
    #         a list containing following:
    #         - Y_gen: the predicted markers
    #         - Yb_gen: the predicted-and-regressed body parameters. Rotations in axis-angle

    #     Raises:
    #         None
    #     """
    #     t_pred = 10-X.shape[0]
    #     Y_gen = self.decode(X, z, t_pred)
    #     print("z.shape = {}".format(z.shape))
    #     # Y_gen = self.predictor.sample_prior(X, z)
    #     # print("regressor_batch_size".format(self.regressor.))
    #     nt, nb = Y_gen.shape[:2]
    #     Yb_gen = self.regressor(Y_gen.view(nt*nb, -1), betas.view(nt*nb, -1))
    #     Yb_gen = Yb_gen.view(nt, nb, -1)
    #     return Y_gen, Yb_gen
    
    def smpl_to_openpose(self,model_type='smplx', use_hands=True, use_face=True,
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

    def get_model2data(self):
        return self.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True, use_face_contour=False, openpose_format='coco25')

    def get_body_model(self,model_type='smplx', gender='female', batch_size=1, device='cpu',joint_mapper=None):
        '''
        type: smpl, smplx smplh and others. Refer to smplx tutorial
        gender: male, female, neutral
        batch_size: an positive integar
        '''
        # body_model_path = '/home/yzhang/body_models/VPoser'
        body_model_path = '/home/yuxinyao/body_models'
        body_model = smplx.create(body_model_path, model_type=model_type,
                                        gender=gender, ext='pkl',
                                        joint_mapper = joint_mapper,
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
                                        ).to(device)
        return body_model

    def points_coord_trans(self,xyz_source_coord, trans_mtx):
        # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
        xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
        xyz_target_coord = xyz_target_coord + trans_mtx[:3, 3].reshape(1,-1)
        return xyz_target_coord

    def points_coord_transT(self,xyz_source_coord, trans_mtx):
        # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
        # print("xyz_source_coord.shape:{}".format(xyz_source_coord.shape))
        # print("trans_mtx.shape:{}".format(trans_mtx.shape))
        xyz_target_coord = torch.matmul(xyz_source_coord,trans_mtx[:3,:3].T)
        xyz_target_coord = xyz_target_coord + trans_mtx[:3, 3].reshape(1,-1)

        # xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
        # xyz_target_coord = xyz_target_coord + trans_mtx[:3, 3].reshape(1,-1)
        # print(xyz_target_coord)
        return xyz_target_coord

    def create_camera(self,camera_type='persp', **kwargs):
    # if camera_type.lower() == 'persp':
    #     return PerspectiveCamera(**kwargs)
        if camera_type.lower() == 'persp_holo':
            return PerspectiveCamera_holo(**kwargs)
        else:
            raise ValueError('Uknown camera type: {}'.format(camera_type))

    def rotate_back(self,bparam,transf_rotate_xaxis,transf_transl_xaxis,delta_T_rotate,transf_rotmat,transf_transl,delta_T):

        # transl_save = bparam['transl'].detach().cpu().numpy()
        # global_ori_save = bparam['global_orient'].detach().cpu().numpy()
        # transl_rotate_recover  = ((np.linalg.inv(transf_rotmat.T))@((transl_save+delta_T[2:]).T)).T+transf_transl-delta_T[2:]
        # global_ori_rotate_recover = R.from_matrix(np.einsum('ij,tjk->tik', np.linalg.inv(transf_rotmat.T),R.from_rotvec(global_ori_save).as_matrix())).as_rotvec()

        # transl_seq_recover = ((np.linalg.inv(transf_rotate_xaxis.T))@((transl_rotate_recover+delta_T_rotate[2:]).T)).T +transf_transl_xaxis-delta_T_rotate[2:]
        # global_orient_seq_recover = R.from_matrix(np.einsum('ij,tjk->tik', np.linalg.inv(transf_rotate_xaxis.T),R.from_rotvec(global_ori_rotate_recover).as_matrix())).as_rotvec()


        transl_save = bparam['transl']
        global_ori_save = bparam['global_orient']

        transf_transl = torch.Tensor(transf_transl).detach().cuda()
        delta_T  = torch.Tensor(delta_T).detach().cuda()
        transf_rotmat = torch.Tensor(transf_rotmat).detach().cuda()

        transf_rotate_xaxis = torch.Tensor(transf_rotate_xaxis).detach().cuda()
        delta_T_rotate = torch.Tensor(delta_T_rotate).detach().cuda()
        transf_transl_xaxis = torch.Tensor(transf_transl_xaxis).detach().cuda() 

        temp_transl = transl_save+delta_T[2:]
        # print("temPp_transl:{},type:{}".format(temp_transl.shape,type(temp_transl)))
        transl_rotate_recover  = torch.matmul((torch.linalg.inv(transf_rotmat.T)),((transl_save+delta_T[2:]).T)).T+transf_transl-delta_T[2:]
        # global_ori_recover = R.from_matrix(np.einsum('ij,tjk->tik', np.linalg.inv(transf_rotmat.T),R.from_rotvec(global_ori_save).as_matrix())).as_rotvec()
        temp = torch.einsum('ij,tjk->tik', torch.linalg.inv(transf_rotmat.T),tgm.angle_axis_to_rotation_matrix(global_ori_save)[:,:3,:3])
        global_orient_rotate_recover = tgm.rotation_matrix_to_angle_axis(F.pad(temp, [0,1])).contiguous().view(-1,3)

        transl_seq_recover = torch.matmul((torch.linalg.inv(transf_rotate_xaxis.T)),((transl_rotate_recover+delta_T_rotate[2:]).T)).T+transf_transl_xaxis-delta_T_rotate[2:]
        temp = torch.einsum('ij,tjk->tik', torch.linalg.inv(transf_rotate_xaxis.T),tgm.angle_axis_to_rotation_matrix(global_orient_rotate_recover)[:,:3,:3])
        global_orient_seq_recover = tgm.rotation_matrix_to_angle_axis(F.pad(temp, [0,1])).contiguous().view(-1,3)
        return transl_seq_recover,global_orient_seq_recover


    def joint2d_loss(self, Y_pred,Yb_pred,kp,cur_fx,cur_fy,holo_cx,holo_cy,cur_world2pv_transform, trans_kinect2holo,gender,betas,transf_rotmat = None,transf_transl= None,delta_T= None,transf_rotate_xaxis= None,
                transf_transl_xaxis= None, delta_T_rotate= None):
        #find 3d key points for predicted y_pred and yb_pred
        joint_mapper = JointMapper(self.get_model2data())
        # print("Yb.shape:{}".format(Yb_pred.shape))
        Yb = Yb_pred.permute(1,0,2)
        Y = Y_pred.permute(1,0,2)
        # print("Yb.shape",Yb.shape)
        bm = self.get_body_model(model_type='smplx',gender = gender,batch_size=8,device='cuda:0',joint_mapper =joint_mapper)
        
        bm2 = self.get_body_model(model_type='smplx', gender=gender,batch_size=8,device='cuda:0')
        batch_joint = np.zeros((3,8,25,2))
        batch_joint_3d = np.zeros((3,8,25,3))

        jts_test = np.zeros((3,8,25,3))
        for i in range(Yb.shape[0]): #i = 0,1,2,-> 3 gens 
            bparam={}
            bparam['transl'] = Yb[i,:,:3]
            bparam['global_orient'] = Yb[i,:,3:6]
            bparam['betas'] = betas[:,i,:]
            bparam['body_pose'] = Yb[i,:,6:69]
            bparam['left_hand_pose'] = Yb[i,:,69:81]
            bparam['right_hand_pose'] = Yb[i,:,81:]
            # jts = bm(return_verts=True, **bparam).joints.detach().cpu().numpy() #[t,J, 3]

            transl, global_orient = self.rotate_back(bparam,transf_rotate_xaxis,transf_transl_xaxis,delta_T_rotate,transf_rotmat,transf_transl,delta_T)
            
            testbp = {}

            testbp['transl'] = torch.Tensor(transl).to('cuda:0')
            testbp['global_orient'] = torch.Tensor(global_orient).to('cuda:0')
            testbp['betas'] = betas[:,i,:]
            testbp['body_pose'] = bparam['body_pose']
            testbp['left_hand_pose'] = bparam['left_hand_pose']
            testbp['right_hand_pose'] = bparam['right_hand_pose']

            jts2 = bm(return_verts=True, **testbp).joints.detach().cpu().numpy() #[t,J, 3]


            joints_2d = np.zeros((8,25,2))
            joints_3d = np.zeros((8,25,3))
            for t in range(2,jts2.shape[0]+2):
                kp_t = kp[t]
                cur_fx_t = cur_fx[t]
                cur_fy_t = cur_fy[t]
                holo_cx_t = holo_cx[t]
                holo_cy_t = holo_cy[t]
                cur_world2pv_transform_t = cur_world2pv_transform[t]
                # cur_world2pv_transform_t = torch.Tensor(cur_world2pv_transform_t).to('cuda:0')
                joints = self.points_coord_trans(jts2[t-2],trans_kinect2holo)
                joints = self.points_coord_trans(joints,cur_world2pv_transform_t)

                add_trans = np.array([[1.0, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])  # different y/z axis definition in opencv/opengl convention
                joints = self.points_coord_trans(joints, add_trans)  # gt 3d joints in current hololens PV(RGB) camera coord, [n_joints, 3]

                camera_center_holo = torch.tensor([holo_cx_t, holo_cy_t]).view(-1, 2)
                device = 'cuda:0'
                camera_holo_kp = self.create_camera(camera_type='persp_holo',
                                                focal_length_x=torch.tensor([cur_fx_t]).to(device).unsqueeze(0),
                                                focal_length_y=torch.tensor([cur_fy_t]).to(device).unsqueeze(0),
                                                center=camera_center_holo,
                                                batch_size=1).to(device=device)
                joints = torch.from_numpy(joints).float().to(device).unsqueeze(0)
                gt_joints_2d = camera_holo_kp(joints)  
                gt_joints_2d = gt_joints_2d.squeeze().detach().cpu().numpy()[0:25]
                
                mask = np.ones_like(gt_joints_2d)
                for ind in range(0, len(kp_t)):
                    if kp_t[ind,0] == 0 and kp_t[ind,1] == 0:
                        mask[ind,:] = [0,0]
                joints_2d[t-2,:,:] = gt_joints_2d*mask
            batch_joint[i,:,:,:] = joints_2d
        kp_3 = np.tile(kp[2:,:,:2],(3,1,1)).reshape((batch_joint.shape))
        # kp_3 = torch.Tensor(kp_3).to('cuda:0').requires_grad_
        return F.mse_loss(torch.Tensor(kp_3),torch.Tensor(batch_joint))

    def joint2d_loss_T(self, Y_pred,Yb_pred,kp,cur_fx,cur_fy,holo_cx,holo_cy,cur_world2pv_transform, trans_kinect2holo,gender,betas,transf_rotmat = None,transf_transl= None,delta_T= None,transf_rotate_xaxis= None,
                transf_transl_xaxis= None, delta_T_rotate= None):
        #find 3d key points for predicted y_pred and yb_pred
        joint_mapper = JointMapper(self.get_model2data())
        # print("Yb.shape:{}".format(Yb_pred.shape))
        Yb = Yb_pred.permute(1,0,2)
        Y = Y_pred.permute(1,0,2)
        # print("Yb.shape",Yb.shape)
        bm = self.get_body_model(model_type='smplx',gender = gender,batch_size=8,device='cuda:0',joint_mapper =joint_mapper)
        
        bm2 = self.get_body_model(model_type='smplx', gender=gender,batch_size=8,device='cuda:0')
        batch_joint = torch.zeros(3,8,25,2).cuda()

        jts_test = torch.zeros(3,8,25,3)
        for i in range(Yb.shape[0]): #i = 0,1,2,-> 3 gens 
            bparam={}
            bparam['transl'] = Yb[i,:,:3]
            bparam['global_orient'] = Yb[i,:,3:6]
            bparam['betas'] = betas[:,i,:]
            bparam['body_pose'] = Yb[i,:,6:69]
            bparam['left_hand_pose'] = Yb[i,:,69:81]
            bparam['right_hand_pose'] = Yb[i,:,81:]
            # jts = bm(return_verts=True, **bparam).joints

            transl, global_orient = self.rotate_back(bparam,transf_rotate_xaxis,transf_transl_xaxis,delta_T_rotate,transf_rotmat,transf_transl,delta_T)
            
            testbp = {}

            testbp['transl'] = torch.Tensor(transl).to(device='cuda:0')
            testbp['global_orient'] = torch.Tensor(global_orient).to(device='cuda:0')
            testbp['betas'] = betas[:,i,:]
            testbp['body_pose'] = bparam['body_pose']
            testbp['left_hand_pose'] = bparam['left_hand_pose']
            testbp['right_hand_pose'] = bparam['right_hand_pose']

            jts2 = bm(return_verts=True, **testbp).joints


            joints_2d = torch.zeros(8,25,2).cuda()
            # joints_3d = np.zeros((8,25,3))
            kpT = torch.Tensor(kp).to(device='cuda:0')
            cur_fxT = torch.Tensor(cur_fx).to(device='cuda:0')
            cur_fyT = torch.Tensor(cur_fy).to(device='cuda:0')
            holo_cxT = torch.Tensor(holo_cx).to(device='cuda:0')
            holo_cyT = torch.Tensor(holo_cy).to(device='cuda:0')
            cur_world2pv_transformT = torch.Tensor(cur_world2pv_transform).to(device='cuda:0')
            trans_kinect2holoT = torch.Tensor(trans_kinect2holo).to(device='cuda:0')


            for t in range(2,jts2.shape[0]+2):
                kp_t = kpT[t]
                cur_fx_t = cur_fxT[t]
                cur_fy_t = cur_fyT[t]
                holo_cx_t = holo_cxT[t]
                holo_cy_t = holo_cyT[t]
                cur_world2pv_transform_t = cur_world2pv_transformT[t]
                # cur_world2pv_transform_t = torch.Tensor(cur_world2pv_transform_t).to('cuda:0')
                joints = self.points_coord_transT(jts2[t-2],trans_kinect2holoT)
                joints = self.points_coord_transT(joints,cur_world2pv_transform_t)

                add_trans = torch.Tensor([[1.0, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]]).cuda()  # different y/z axis definition in opencv/opengl convention
                joints = self.points_coord_transT(joints, add_trans)  # gt 3d joints in current hololens PV(RGB) camera coord, [n_joints, 3]

                camera_center_holo = torch.tensor([holo_cx_t, holo_cy_t]).view(-1, 2)
                device = 'cuda:0'
                camera_holo_kp = self.create_camera(camera_type='persp_holo',
                                                focal_length_x=torch.tensor([cur_fx_t]).to(device).unsqueeze(0),
                                                focal_length_y=torch.tensor([cur_fy_t]).to(device).unsqueeze(0),
                                                center=camera_center_holo,
                                                batch_size=1).to(device=device)
                joints =joints.unsqueeze(0)  
                # print("gt_joints_2d.shape:{}".format(joints.shape))
                gt_joints_2d = camera_holo_kp(joints)  
                gt_joints_2d = gt_joints_2d.squeeze()[0:25]
                
                mask = torch.ones_like(gt_joints_2d)
                for ind in range(0, len(kp_t)):
                    if kp_t[ind,0] == 0 and kp_t[ind,1] == 0:
                        mask[ind,:] = torch.Tensor([0,0])
                joints_2d[t-2,:,:] = gt_joints_2d*mask
            batch_joint[i,:,:,:] = joints_2d

        # print("kp.shape",kp.shape)
        # print("batch_joint.shape",batch_joint.shape)
        # print(kp[2:,:,:2].repeat(3,1).reshape(batch_joint.shape).shape)

        kp_3 = kpT[2:,:,:2].repeat(3,1,1).reshape(batch_joint.shape)
        # kp_3 = torch.Tensor(kp_3).to('cuda:0').requires_grad_
        # print(type(kp_3))
        # print("kp_3.shape",kp_3.shape)
        # print("batch_joint.shape",batch_joint.shape)

        # print(kp[2:,:,:2].repeat(3,1).reshape(batch_joint.shape).shape)
        return F.mse_loss(kp_3,batch_joint)

        # return (np.square(kp_3 - batch_joint)).mean(axis=1)
        # return (joints**2).mean(axis=1)

    def recover(self, 
                data: Union[torch.Tensor, np.ndarray],
                bparams: Union[torch.Tensor, np.ndarray],
                betas: Union[torch.Tensor, np.ndarray],
                n_gens: int=-1,
                to_numpy: bool=True,
                param_blending: bool=True,
                t_his = None,
                lr = 0.05, cur_world2pv_transform = None,
                cur_fx= None,
                cur_fy = None,
                holo_cx = None,
                holo_cy = None,
                trans_kinect2holo= None,
                kp= None,
                gender = 'female',
                transf_rotmat = None,transf_transl= None,delta_T= None,transf_rotate_xaxis= None,
                transf_transl_xaxis= None, delta_T_rotate= None
            ):
        if type(data) is np.ndarray:
            traj = torch.tensor(data, device=self.device,
                    dtype=self.dtype).contiguous()
        elif type(data) is torch.Tensor:
            traj = data.detach()

        if type(bparams) is np.ndarray:
            bparams = torch.tensor(bparams, device=self.device,
                    dtype=self.dtype).contiguous()
        elif type(bparams) is torch.Tensor:
            bparams = bparams.detach()

        

        t_his_ = t_his
        t_pred_ = traj.shape[0]-t_his_
        # print("t_pred = {}".format(t_pred_))
        X = traj[:t_his_]
        Xb = bparams[:t_his_]
        if n_gens > 0:
            X = X.repeat((1, n_gens, 1))
            Xb = Xb.repeat((1, n_gens, 1))
            # print("X.shape:{}".format(X.shape))
        if type(betas) is np.ndarray: #betas=(10,) or (1,10)
            # print(betas.shape)
            betas = torch.cuda.FloatTensor(np.tile(betas, (t_pred_, X.shape[1], 1)),device=self.device)
            # betas = torch.cuda.FloatTensor(betas,device=self.device)
        elif type(betas) is torch.Tensor: #betas=(1,10) or (10,)
            betas = betas.repeat(t_pred_, X.shape[1], 1)

        z_rec = torch.randn(X.shape[1], self.model.predictor.z_dim,requires_grad=True,device=self.device)
        l = (z_rec**2).mean()
        l.backward()
        optimizer = torch.optim.Adam([z_rec], lr=0.1)
        scheduler = MultiStepLR(optimizer, milestones=[100,200], gamma=0.5)
        Y_pred, Yb_pred = 0,0
        Y_pred_intial, Yb_pred_intial = 0,0
        torch.backends.cudnn.enabled=False
        for iteration in range(300):
            

            stime = time.time()
            Y_pred, Yb_pred = self.model.sample_prior(X, betas=betas, z=z_rec)
            if iteration ==0:
                Y_pred_intial, Yb_pred_intial = Y_pred, Yb_pred
            # print("z.requires_grad:{}".format(z_rec.requires_grad))
            loss_2dkp = self.joint2d_loss_T(Y_pred,Yb_pred,kp,cur_fx,cur_fy,holo_cx,holo_cy,cur_world2pv_transform, trans_kinect2holo,gender,betas[0:8,:],
                                        transf_rotmat,transf_transl,delta_T,transf_rotate_xaxis,transf_transl_xaxis,delta_T_rotate)

            # print("loss_2dkp.require_grad:{}".format(loss_2dkp.requires_grad))
            
            # if iteration%50 ==0:
            #     optimizer.lr = lr/2

            optimizer.zero_grad()
            #lr = 0.25 good
            # print("z.grad2 = {}".format(z_rec.grad))
            z_rec.retain_grad()
            # print("z.requires_grad:{}".format(z_rec.requires_grad))
            loss_2dkp.backward()

            optimizer.step()
            lr_o = optimizer.param_groups[0]['lr']
            print("{}th iter, loss:{},time:{},lr:{}".format(iteration,loss_2dkp.item(),time.time()-stime,lr_o))
            scheduler.step()


            # break

        Y_pred, Yb_pred = self.model.sample_prior(X, betas=betas, z=z_rec)
        
        # plot()


        # print("Y_pred.shape:{}".format(Y_pred.shape))
        Y = torch.cat((X[:,:,:147*3], Y_pred), dim=0)
        Yb = torch.cat((Xb, Yb_pred), dim=0)
        Y_pred_intial = torch.cat((X[:,:,:147*3], Y_pred_intial), dim=0)
        Yb_pred_intial = torch.cat((Xb, Yb_pred_intial), dim=0)
        if param_blending:
            Yb = self._blend_params(Yb)
            Y_pred_intial = self._blend_params(Y_pred_intial)
        # from [t,b,d] to [b,t,d]
        Y = Y.permute(1,0,2)
        Y_pred_intial = Y_pred_intial.permute(1,0,2)
        Yb = Yb.permute(1,0,2)
        Yb_pred_intial = Yb_pred_intial.permute(1,0,2)
        # print("Y.shape:{}".format(Y.shape))

        Y = Y.detach().cpu().numpy()
        Y_pred_intial = Y_pred_intial.detach().cpu().numpy()
        Yb = Yb.detach().cpu().numpy()
        Yb_pred_intial = Yb_pred_intial.detach().cpu().numpy()
        # z = z_rec.detach().cpu().numpy()
        return Y, Yb,Y_pred_intial, Yb_pred_intial





    def _compose_body_params_(self, data):
        # print(data.keys())
        # print(data)
        import json
        # with open('./before_compose.txt', 'w') as file:
        #     # file.write(json.dumps(data)) 
        #     print('pose.shape:', data['poses'].shape)
        #     print('bodypose.shape:', data['body_pose'].shape)
        #     file.write(str(data))
        transl = data['transl']
        glorot = data['glorot']
        # print("transl.shape:{}, glorot.shape:{}".format(transl.shape, glorot.shape))
        body_pose = data['poses'][:,:63]
        hand_pose = np.zeros((transl.shape[0], 24))
        xb = np.concatenate([transl, glorot, body_pose, hand_pose],axis=-1) #[t,d]
        return xb
    def _blend_params(self, body_params, t_his=None):
        # Yb = torch.cat((Xb, Yb_pred), dim=0)
        if t_his is None:
            t_his = self.t_his
        param_n = body_params[t_his-1, :, 6:]
        param_p = body_params[t_his+1, :, 6:]
        body_params[t_his, :, 6:] = (param_n+param_p)/2.0

        t_his = t_his+1
        param_n = body_params[t_his-1, :, 6:]
        param_p = body_params[t_his+1, :, 6:]
        body_params[t_his, :, 6:] = (param_n+param_p)/2.0
        return body_params
    def recover_primitive_to_files(self, batch_gen, n_seqs, n_gens, t_his=None):
        '''
        n_seqs: how many sequences to generate
        n_gens: for each input sequence, how many different sequences to predict
        '''
        # self.build_model()
        ### generate data and save them to files. They will need inverse kinematics to get body mesh.
        ### generate data
        gen_results = {}
        gen_results['gt'] = []
        gen_results['betas'] = []
        gen_results['gender'] = []
        gen_results['markers'] = []
        gen_results['smplx_params'] = []
        gen_results['markers_before'] = []
        gen_results['smplx_params_before'] = []
        gen_results['transl'] = []
        gen_results['bparams_seed'] = []
        gen_results['kp'] = []
        gen_results['cur_world2pv_transform'] = []
        gen_results['camera_holo_kp'] = []
        gen_results['trans_kinect2holo'] = []
        gen_results['cur_fx'] = []
        gen_results['cur_fy'] = []
        gen_results['holo_cx'] = []
        gen_results['holo_cy'] = []
        gen_results['batch_joint'] = []
        gen_results['address'] = []

        gen_results['transf_rotmat']  = []
        gen_results['transf_transl']  = []
        gen_results['delta_T'] = []
        gen_results['transf_rotate_xaxis'] = []
        gen_results['transf_transl_xaxis'] = []
        gen_results['delta_T_rotate'] = []
        idx = 0
        while idx < n_seqs:
            print('[INFO] generating with motion seed {}'.format(idx))
            data = batch_gen.next_sequence()
            if str(data['gender']) != self.regressorcfg.modelconfig['gender']:
                continue
            #body_feature = joints
            motion_np = data['body_feature']
            motion = torch.cuda.FloatTensor(motion_np).unsqueeze(1) #[t,b,d]
            bparams_np = self._compose_body_params_(data)
            bparams = torch.cuda.FloatTensor(bparams_np).unsqueeze(1) #[t,b,d]
            if (self.predictorcfg.modelconfig['body_repr'] == 'ssm2_67_grab'):
                gen_results['gt'].append(motion_np[:,:147*3].reshape((1,motion_np.shape[0],-1,3)))
            else:
                gen_results['gt'].append(motion_np[:,:67*3].reshape((1,motion_np.shape[0],-1,3)))
            gen_results['betas'].append(data['betas'])
            gen_results['gender'].append(str(data['gender']))
            gen_results['transf_rotmat'].append(data['transf_rotmat'])
            gen_results['transf_transl'].append(data['transf_transl'])




            gen_results['transl'].append(data['transl'][None,...]) #[b,t,d]
            gen_results['bparams_seed'].append(bparams_np)
            gen_results['cur_world2pv_transform'].append(data['cur_world2pv_transform'])
            # gen_results['camera_holo_kp'].append(data['camera_holo_kp'])
            gen_results['cur_fx'].append(data['cur_fx'])
            gen_results['cur_fy'].append(data['cur_fy'])
            gen_results['holo_cx'].append(data['holo_cx'])
            gen_results['holo_cy'].append(data['holo_cy'])

            gen_results['trans_kinect2holo'].append(data['trans_kinect2holo'])
            # print("data.keys:{}".format(data.keys()))
            gen_results['kp'].append(data['kp2d'])

            gen_results['transf_rotmat'].append(data['transf_rotmat'])
            gen_results['transf_transl'].append(data['transf_transl'])
            gen_results['delta_T'].append(data['delta_T'])
            gen_results['transf_rotate_xaxis'].append(data['transf_rotate_xaxis'])
            gen_results['transf_transl_xaxis'].append(data['transf_transl_xaxis'])
            gen_results['delta_T_rotate'].append(data['delta_T_rotate'])
            # for key in data.keys():
            #     if key =='gender':
            #         continue
            #     print("gen_result[{}].shape:{}".format(key, data[key].shape))
            # generate optimized marker& body param
            # print("data['camera_holo_kp'])", len(data['camera_holo_kp']))
            transf_rotmat =data['transf_rotmat']
            transf_transl =data['transf_transl']
            delta_T=data['delta_T']
            transf_rotate_xaxis=data['transf_rotate_xaxis']
            transf_transl_xaxis=data['transf_transl_xaxis']
            delta_T_rotate=data['delta_T_rotate']
            pred_markers, pred_body_params,pred_marker_before,pred_body_params_before= self.recover(motion, bparams, betas=data['betas'][0,:],
                                                        n_gens=n_gens, t_his=t_his, cur_world2pv_transform = data['cur_world2pv_transform'],
                                                        # camera_holo_kp = data['camera_holo_kp'],
                                                        cur_fx = data['cur_fx'],
                                                        cur_fy = data['cur_fy'],
                                                        holo_cx = data['holo_cx'],
                                                        holo_cy = data['holo_cy'],
                                                        trans_kinect2holo = data['trans_kinect2holo'],
                                                        kp = data['kp2d'],
                                                        gender = str(data['gender']),
                                                        transf_rotmat=transf_rotmat,transf_transl=transf_transl,delta_T=delta_T,transf_rotate_xaxis=transf_rotate_xaxis,
                                                        transf_transl_xaxis=transf_transl_xaxis, delta_T_rotate=delta_T_rotate

                                                        )
            pred_markers = np.reshape(pred_markers, (pred_markers.shape[0], pred_markers.shape[1],-1,3))
            pred_marker_before = np.reshape(pred_marker_before, (pred_marker_before.shape[0], pred_marker_before.shape[1],-1,3))
            # pred_markers = np.repeat(data['kp2d'],3,axis=0).reshape(3,10,25,3)
            # gen_results['batch_joint'].append(batch_joint)
            # print('pred_markers.shape:{}'.format(pred_markers.shape))
            gen_results['markers'].append(pred_markers)
            gen_results['smplx_params'].append(pred_body_params)
            gen_results['markers_before'].append(pred_marker_before)
            gen_results['smplx_params_before'].append(pred_body_params_before)
            gen_results['address'].append(data['address'])
            idx+=1
        gen_results['gt'] = np.stack(gen_results['gt'])
        # print("gen_results['gt'].shape:{}".format(gen_results['gt'].shape))
        gen_results['markers'] = np.stack(gen_results['markers']) #[#seq, #genseq_per_pastmotion, t, #joints, 3]
        gen_results['smplx_params'] = np.stack(gen_results['smplx_params'])
        gen_results['markers_before'] = np.stack(gen_results['markers_before']) #[#seq, #genseq_per_pastmotion, t, #joints, 3]
        gen_results['smplx_params_before'] = np.stack(gen_results['smplx_params_before'])
        # print('smplx_params.shape:{}'.format(gen_results['smplx_params'].shape))
        gen_results['transl'] = np.stack(gen_results['transl'])
        # gen_results['address'] = np.stack(gen_results['address'])
        # gen_results['batch_joint'] =np.stack(gen_results['batch_joint'])
        # print("batchjoint.shape:{}".format(gen_results['batch_joint'].shape))
        ### save to file
        outfilename = os.path.join(
                            self.testconfig['result_dir'],
                            'mp_gen_seed{}'.format(self.testconfig['seed']),
                            batch_gen.amass_subset_name[0]
                        )
        if not os.path.exists(outfilename):
            os.makedirs(outfilename)
        outfilename = os.path.join(outfilename,
                        '1results_{}_{}.pkl'.format(self.predictorcfg.modelconfig['body_repr'],
                                                self.regressorcfg.modelconfig['gender']
                                                )
                        )
        with open(outfilename, 'wb') as f:
            pickle.dump(gen_results, f)
        print("gen_results saved to {}".format(outfilename))
        
