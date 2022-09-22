import cv2
import numpy as np
import glob


img_array = []
i = 0
t = 10
# for filename in sorted(glob.glob('/home/yuxinyao/datasets/egobody/canicalized-camera-wearer-grab/recording_20210910_S06_S05_01/subseq_*.pkl_renderbody/seq0_gen0/*.png')):
# print(sorted(glob.glob('/home/yuxinyao/GAMMA-release/results/exp_GAMMAPrimitive/MPVAE_2frame_grab_v4/results/mp_gen_seed0/canicalized-camera-wearer-grab-openpose/1results_ssm2_67_grab_female.pkl_renderbody_image/seq0_gen0_after/*.png')))
for filename in sorted(glob.glob('/home/yuxinyao/GAMMA-release/results/exp_GAMMAPrimitive/MPVAE_2frame_grab_v4/results/mp_gen_seed0/canicalized-camera-wearer-grab-openpose/1results_ssm2_67_grab_female.pkljustTest_renderbody_ppt/seq*_gen0_after_2d/*.jpg')):
    # print("filename:",filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/yuxinyao/GAMMA-release/results/exp_GAMMAPrimitive/MPVAE_2frame_grab_v4/results/mp_gen_seed0/canicalized-camera-wearer-grab-openpose/1results_ssm2_67_grab_female.pkljustTest_renderbody_ppt/after.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 5, size)
#  print()
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()