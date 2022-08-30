import os
import sys
import numpy as np
import open3d as o3d
import torch
import smplx
import cv2
import pickle
import pdb
import re
import glob
import trimesh
import pyrender
import PIL.Image as pil_img
sys.path.append(os.getcwd())
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized
from exp_GAMMAPrimitive.utils.vislib import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


t = 10


def visualize(data, data_bparams, datagt=None,joints= None, gender='male', betas=0,
                outfile_path=None, datatype='gt',seq=0,gen=0,
                show_body=True,string='after',cur_world2pv_transform_t=None, trans_kinect2holo_t=None):
    ## prepare data
    n_frames = t

    ## prepare visualizer
    np.random.seed(0)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=540,visible=True)
    # vis.create_window(width=480, height=270,visible=True)
    render_opt=vis.get_render_option()
    render_opt.mesh_show_back_face=True
    render_opt.line_width=10
    render_opt.point_size=5
    render_opt.background_color = color_hex2rgb('#1c2434')
    vis.update_renderer()

    ### top lighting
    box = o3d.geometry.TriangleMesh.create_box(width=200, depth=1,height=200)
    box.translate(np.array([-200,-200,6]))
    vis.add_geometry(box)
    vis.poll_events()
    vis.update_renderer()

    #### world coordinate
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    vis.add_geometry(coord)
    vis.poll_events()
    vis.update_renderer()

    ## create body mesh from data
    ball_list = []
    ball_list2 = []
    # n_markers = int(re.split('_|\.',
    #                 os.path.basename(results_file_name))[2])
    n_markers = len(data[1])

    for i in range(n_markers):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        vis.add_geometry(ball)
        vis.poll_events()
        vis.update_renderer()
        ball_list.append(ball)

    for i in range(25):
        ball2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        vis.add_geometry(ball2)
        vis.poll_events()
        vis.update_renderer()
        ball_list2.append(ball2)

    if datagt is not None:
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        pcd.points = o3d.utility.Vector3dVector(datagt[-1])
        pcd.paint_uniform_color(color_hex2rgb('#dce04a')) # "I want hue"
        vis.update_geometry(pcd)

    if show_body:
        body = o3d.geometry.TriangleMesh()
        vis.add_geometry(body)
        vis.poll_events()
        vis.update_renderer()

    ## parse smplx parameters
    print(gender)
   
    ## parse smplx parameters
    bm = get_body_model('smplx', gender, n_frames, device='cpu')
    # print('betas.shape:{}'.format(betas.shape))
    print("data_bprams.shape:{}".format(data_bparams.shape))
    bparam = {}
    bparam['transl'] = data_bparams[:,:3]
    bparam['global_orient'] = data_bparams[:,3:6]
    bparam['betas'] = np.tile(betas[None,...], (n_frames,1))
    bparam['body_pose'] = data_bparams[:,6:69]
    bparam['left_hand_pose'] = data_bparams[:,69:81]
    bparam['right_hand_pose'] = data_bparams[:,81:]
    # print('transl.shape:{}'.format(bparam['transl'].shape))


    ## obtain body mesh sequences
    for key in bparam:
        bparam[key] = torch.FloatTensor(bparam[key])
        print('{}:{}'.format(key, bparam[key].shape))
    # tem_bm = bm(return_verts=True, **bparam).apply_transforms(trans_kinect2holo_t).apply_transforms(cur_world2pv_transform_t)
    verts_seq = bm(return_verts=True, **bparam).vertices.detach().cpu().numpy() #[t,verts, 3]
    jts = bm(return_verts=True, **bparam).joints[:,:22].detach().cpu().numpy() #[t,J, 3]
    
    frame_idx = 0
    cv2.namedWindow('frame2')
    for it in range(0,n_frames):
        for i,b in enumerate(ball_list):
            # print("data.shape{}".format(data.shape))
            b.translate(data[it,i], relative=False)
            vis.update_geometry(b)


        if it <motion_seed_len:
            for ball in ball_list:
                ball.paint_uniform_color(color_hex2rgb('#c2dd97')) # "I want hue"
        else:
            if datatype == 'gt':
                pass
            else:
                for ball in ball_list:
                    ball.paint_uniform_color(color_hex2rgb('#c7624f')) # "I want hue"
        if show_body:
            ## set body mesh locations
            body.vertices = o3d.utility.Vector3dVector(verts_seq[it])
            body.triangles = o3d.utility.Vector3iVector(bm.faces)
            body.vertex_normals = o3d.utility.Vector3dVector([])
            body.triangle_normals = o3d.utility.Vector3dVector([])
            body.compute_vertex_normals()
            vis.update_geometry(body)

            if it <motion_seed_len:
                body.paint_uniform_color(color_hex2rgb('#c2dd97')) # "I want hue"
            else:
                body.paint_uniform_color(color_hex2rgb('#c7624f')) # "I want hue"
        # if it >=2:
        if joints is not None:
            for i,b in enumerate(ball_list2):
                b.translate(joints[it,i], relative=False)
                vis.update_geometry(b)
        # for i,b in enumerate(ball_list2):
        #     # print(type(joints[it,i]))
        #     # print(joints[it,i].shape)
        # # print("data.shape{}".format(data.shape))
        #     b.translate(joints[it,i], relative=False)
        #     vis.update_geometry(b)

        ## update camera.
        ctr = vis.get_view_control()
        ctr.set_constant_z_far(10)
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        ### get cam T
        # body_t = np.mean(data[it],axis=0) # let cam follow the body
        body_t = np.array([0,0,0])
        # body_t = -data_bparams['trans'][it]
        cam_t = body_t + 2.0*np.ones(3)
        ### get cam R
        cam_z =  body_t - cam_t
        cam_z = cam_z / np.linalg.norm(cam_z)
        cam_x = np.array([cam_z[1], -cam_z[0], 0.0])
        cam_x = cam_x / np.linalg.norm(cam_x)
        cam_y = np.array([cam_z[0], cam_z[1], -(cam_z[0]**2 + cam_z[1]**2)/cam_z[2] ])
        cam_y = cam_y / np.linalg.norm(cam_y)
        cam_r = np.stack([cam_x, -cam_y, cam_z], axis=1)
        ### update render cam
        transf = np.eye(4)
        transf[:3,:3]=cam_r
        transf[:3,-1] = cam_t
        cam_param = update_render_cam(cam_param, transf)
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()

        ## capture RGB appearance
        rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        cv2.imshow("frame2", np.uint8(255*rgb[:,:,[2,1,0]]))
        if outfile_path is not None:
            renderimgname = os.path.join(outfile_path, 'img_{:05d}_{}.png'.format(frame_idx,string))
            frame_idx = frame_idx + 1
            cv2.imwrite(renderimgname, np.uint8(255*rgb[:,:,[2,1,0]]))
        cv2.waitKey(5)

def visualize2d(data, data_bparams, address, cur_fx,cur_fy,holo_cx,holo_cy, gender='male', betas=0,
                outfile_path=None, datatype='gt',seq=0,gen=0,string = 'after',trans_kinect2holo_t = None,cur_world2pv_transform_t= None):
    body_model = get_body_model('smplx', gender, t, device='cpu')
    H, W = 1080, 1920
    n_frames = t
    bparam = {}
    bparam['transl'] = data_bparams[:,:3]
    bparam['global_orient'] = data_bparams[:,3:6]
    bparam['betas'] = np.tile(betas[None,...], (n_frames,1))
    bparam['body_pose'] = data_bparams[:,6:69]
    bparam['left_hand_pose'] = data_bparams[:,69:81]
    bparam['right_hand_pose'] = data_bparams[:,81:]
    # print('transl.shape:{}'.format(bparam['transl'].shape))

    bparam['transl'], bparam['global_orient'] = rotate_back(bparam,transf_rotate_xaxis,transf_transl_xaxis,delta_T_rotate,transf_rotmat,transf_transl,delta_T,False)
    ## obtain body mesh sequences
    for key in bparam:
        bparam[key] = torch.FloatTensor(bparam[key])
        print('{}:{}'.format(key, bparam[key].shape))
    

    verts_seq = body_model(return_verts=True, **bparam).vertices.detach().cpu().numpy() #[t,verts, 3]
    # output = body_model(return_verts=True, **torch_param)
    # vertices = output.vertices.detach().cpu().numpy().squeeze()
    for i in range(t):
        address_t = address[i]
        cur_fx_t = cur_fx[i]
        cur_fy_t = cur_fy[i]
        holo_cx_t = holo_cx[i]
        holo_cy_t = holo_cy[i]
        body = trimesh.Trimesh(verts_seq[i,:,:], body_model.faces, process=False)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_trisurf(body.vertices[:, 0], body.vertices[:,1], triangles=body.faces, Z=body.vertices[:,2]) 
        # plt.show()

        camera_center = np.array([holo_cx_t, holo_cy_t])
        camera_pose = np.eye(4)
        camera_pose = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1) * camera_pose
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        camera = pyrender.camera.IntrinsicsCamera( #
                    fx=cur_fx_t, fy=cur_fy_t,
                    cx=camera_center[0], cy=camera_center[1])
        # print("cur_world2pv_transform_tshape:{}".format(cur_world2pv_transform_t.shape))
        body.apply_transform(trans_kinect2holo_t)  # master kinect RGB coord --> hololens world coord
        body.apply_transform(cur_world2pv_transform_t[i])  # hololens world coord --> current frame hololens pv(RGB) coordinate
        base_color = (1.0, 193/255, 193/255, 1.0)
        material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=base_color
        )
        body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
        img = cv2.imread(address_t)[:, :, ::-1]
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(body_mesh, 'mesh')
        r = pyrender.OffscreenRenderer(viewport_width=W,
                                        viewport_height=H,
                                        point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

        color = color.astype(np.float32) / 255.0
        alpha = 1.0  # set transparency in [0.0, 1.0]
        color[:, :, -1] = color[:, :, -1] * alpha
        color = pil_img.fromarray((color * 255).astype(np.uint8))
        output_img = pil_img.fromarray((img[50:450,300:650]).astype(np.uint8))
        output_img.paste(color, (0, 0), color)
        output_img.convert('RGB')
        # output_img = output_img.resize((int(W / 2), int(H / 2)))
        # print(output_img.size)
        output_img.save(os.path.join(outputfolder, 'holo_' + str(i) + '_output.jpg'))
    return 0



if __name__=='__main__':
    proj_path = os.getcwd()
    exps = [
            'exp_GAMMAPrimitive/MPVAE_1frame_v4',
            ]
    n_seq_vis = 3
    n_gen_vis = 3
    for show_body in [True]:
        out_suffix='body' if show_body else 'kps'
        for exp in exps:
            # res_file_list = sorted(glob.glob(proj_path+'/results/exp_GAMMAPrimitive/MPVAE_2frame_grab_v4/results/mp_gen_seed0/canicalized-camera-wearer-grab/results_ssm2_67_grab_female.pkl'))
            # res_file_list = sorted(glob.glob(proj_path+'/results/exp_GAMMAPrimitive/MPVAE_2frame_grab_v4/results/mp_gen_seed0/canicalized-camera-wearer-grab/results_ssm2_67_grab_female.pkl'))
            res_file_list = sorted(glob.glob(proj_path+"/results/exp_GAMMAPrimitive/MPVAE_2frame_grab_v4/results/mp_gen_seed0/canicalized-camera-wearer-grab-openpose/1results_ssm2_67_grab_female.pkl"))
            # res_file_list = sorted(glob.glob( '/home/yuxinyao/datasets/egobody/canicalized-camera-wearer/recording_20210911_S07_S06_01/*.pkl'))
            # res_file_list = sorted(glob.glob( '/home/yuxinyao/datasets/egobody/canicalized-camera-wearer-Face-x1/recording_20210910_S06_S05_01/*.pkl'))
            for results_file_name in res_file_list:
                # print("1")
                print('-- processing: '+results_file_name)
                with open(results_file_name, 'rb') as f:
                    data = pickle.load(f)
                # dd = data['marker_ssm2_67']
                dd = data['markers']
                dd_before = data['markers_before']
                print("marker: {}".format(dd.shape))
                ddgt = data['gt']
                # print("ddgt.shape: {}".format(ddgt.shape))
                dd_bparam = data['smplx_params']
                dd_bparam_before = data['smplx_params_before']
                
                # dd_bparam = data['bparams_seed']
                n_seq=dd.shape[0]
                n_gen=dd.shape[1]
                motion_seed_len = 1
                # joints = data['batch_joint']
                # print("joints.shape: {}".format(joints.shape))

                trans_kinect2holo = data['trans_kinect2holo']
                cur_world2pv_transform = data['cur_world2pv_transform']
                address = data['address']
                cur_fx=(data['cur_fx'])
                cur_fy=(data['cur_fy'])
                holo_cx=(data['holo_cx'])
                holo_cy=(data['holo_cy'])
                print(len(address[0]))
                st = 'after'

                if st=='before':
                    dd = dd_before
                    dd_bparam = dd_bparam_before
                else:
                    pass
                for seq in range(n_seq_vis):
                    gender = data['gender'][seq]
                    betas = data['betas'][seq][0]

                    transf_rotmat =data['transf_rotmat'][seq]
                    transf_transl =data['transf_transl'][seq]
                    delta_T=data['delta_T'][seq]
                    transf_rotate_xaxis=data['transf_rotate_xaxis'][seq]
                    transf_transl_xaxis=data['transf_transl_xaxis'][seq]
                    delta_T_rotate=data['delta_T_rotate'][seq]
                    for gen in range(n_gen_vis):

                        renderfolder = results_file_name+'_render{}_ppt'.format(out_suffix)+'/seq{}_gen{}_{}'.format(seq, gen,st)
                        outputfolder = results_file_name+'_render{}_image_ppt'.format(out_suffix)+'/seq{}_gen{}_{}'.format(seq, gen,st)

                        print(renderfolder)
                        if not os.path.exists(renderfolder):
                            os.makedirs(renderfolder)
                        if not os.path.exists(outputfolder):
                            os.makedirs(outputfolder)
                        # visualize(dd[seq,gen],
                        #         dd_bparam[seq,gen],
                        #         # datagt=ddgt[seq,0], # change this one to none will not visualize the targer marker.
                        #         # datagt = None,
                        #         # joints = joints[seq,gen],
                        #         gender=gender,
                        #         betas=betas,
                        #         outfile_path=renderfolder, datatype='kps',
                        #         seq=seq,gen=gen,
                        #         show_body=True,
                        #         string=st,cur_world2pv_transform_t=cur_world2pv_transform[seq], trans_kinect2holo_t=trans_kinect2holo[seq])
                        visualize2d(dd[seq,gen],
                                dd_bparam[seq,gen],
                                address= address[seq], cur_fx = cur_fx[seq],cur_fy = cur_fy[seq],holo_cx = holo_cx[seq],holo_cy = holo_cy[seq],
                                gender=gender,
                                betas=betas,
                                outfile_path=renderfolder+'_2d', datatype='kps',
                                seq=seq,gen=gen,
                                string=st, cur_world2pv_transform_t=cur_world2pv_transform[seq], trans_kinect2holo_t=trans_kinect2holo[seq]

                        )
