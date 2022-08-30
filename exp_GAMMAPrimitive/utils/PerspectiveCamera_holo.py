import numpy as np
import cv2
import ast
import torch
# import torch.nn as nn
from smplx.lbs import transform_mat
from scipy.spatial.transform import Rotation as R
def get_new_coordinate_rotate(bm_one, transl,global_orient,pose,betas):
    temp_bodyconfig = {}
    if(type(transl) is not torch.Tensor):
        temp_bodyconfig['transl'] = torch.FloatTensor(transl).cuda()
        temp_bodyconfig['global_orient'] = torch.FloatTensor(global_orient).cuda()
        temp_bodyconfig['body_pose'] = torch.FloatTensor(pose).cuda()
        temp_bodyconfig['betas'] = torch.FloatTensor(betas).cuda()

    smplxout = bm_one(**temp_bodyconfig)
    joints = smplxout.joints.squeeze().detach().cpu().numpy()

    global_ori_rotate = R.from_rotvec([-np.pi/2,0,0]).as_matrix()
    transl_new = joints[0,:1,:] # put the local origin to pelvis

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
    if(type(transl) is not torch.Tensor):
        temp_bodyconfig_seq['transl'] = torch.zeros([n_batches,3], dtype=torch.float32).cuda()
        temp_bodyconfig_seq['global_orient'] = torch.zeros([n_batches,3], dtype=torch.float32).cuda()
        temp_bodyconfig_seq['body_pose'] = torch.FloatTensor(pose).cuda()
        temp_bodyconfig_seq['betas'] = torch.FloatTensor(betas).cuda()
    smplx_out = body_mesh_model(return_verts=True, **temp_bodyconfig_seq)
    delta_T = smplx_out.joints[:,0,:] # we output all pelvis locations
    delta_T = delta_T.detach().cpu().numpy() #[t, 3]

    return delta_T

class PerspectiveCamera_holo(torch.nn.Module):

    FOCAL_LENGTH = 5000

    def __init__(self, rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera_holo, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero', torch.zeros([batch_size], dtype=dtype))

        self.register_buffer('focal_length_x', focal_length_x)    # Adds a persistent buffer to the module
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)     # [bs, 2]

        if rotation is None:
            rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)    # [bs, 3, 3]

        rotation = torch.nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)  # Adds a parameter to the module, shape [1,3,3],  [[1,0,0],[0,1,0],[0,0,1]]

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)   # [bs, 3]

        translation = torch.nn.Parameter(translation,
                                   requires_grad=True)
        self.register_parameter('translation', translation)  # all 0

    def forward(self, points):
        device = points.device  # [bs, 118, 3]

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x  # todo, self.focal_length_x: [bs]
            camera_mat[:, 1, 1] = self.focal_length_y  # [bs, 2, 2], each batch: [[f_x, 0], [0, f_y]]

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))   # [bs, 4, 4], each batch: I
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)     # [bs, 118, 1]
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)   # [1, 118, 4]

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])   # [1, 118, 4]

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))  # [1, 118, 2]
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) + self.center.unsqueeze(dim=1)
        return img_points   # [1, 118, 2]



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


    def create_camera(camera_type='persp', **kwargs):
        # if camera_type.lower() == 'persp':
        #     return PerspectiveCamera(**kwargs)
        if camera_type.lower() == 'persp_holo':
            return PerspectiveCamera_holo(**kwargs)
        else:
            raise ValueError('Uknown camera type: {}'.format(camera_type))
