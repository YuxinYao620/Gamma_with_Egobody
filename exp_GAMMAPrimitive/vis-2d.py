            fpv_img_path = os.path.join(fpv_recording_dir, 'PV', holo_frame_id_dict[holo_frame_id])
            cur_fx = holo_fx_dict[holo_frame_id]
            cur_fy = holo_fy_dict[holo_frame_id]
            cur_pv2world_transform = holo_pv2world_trans_dict[holo_frame_id]
            cur_world2pv_transform = np.linalg.inv(cur_pv2world_transform)

            camera = pyrender.camera.IntrinsicsCamera( #
                fx=cur_fx, fy=cur_fy,
                cx=camera_center[0], cy=camera_center[1])

            body.apply_transform(trans_kinect2holo)  # master kinect RGB coord --> hololens world coord
            body.apply_transform(cur_world2pv_transform)  # hololens world coord --> current frame hololens pv(RGB) coordinate
            body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
            # if save_meshes:
            #     body.export(osp.join(meshes_dir,img_name, '000.ply'))

            if args.rendering_mode == 'body' or args.rendering_mode == 'both':
                img = cv2.imread(fpv_img_path)[:, :, ::-1]
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
                output_img = pil_img.fromarray((img).astype(np.uint8))
                output_img.paste(color, (0, 0), color)
