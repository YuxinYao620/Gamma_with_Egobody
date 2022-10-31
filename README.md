# Gamma with Egobody

<p align="center">
  <img width="900" height="auto" src="demo.gif">
</p>

This repo contains the official implementation of paper:

```
@inproceedings{zhang2022wanderings,
  title={The Wanderings of Odysseus in 3D Scenes},
  author={Zhang, Yan and Tang, Siyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20481--20491},
  year={2022}
}
```
```
@inproceedings{Zhang:ECCV:2022,
   title = {EgoBody: Human Body Shape and Motion of Interacting People from Head-Mounted Devices},
   author = {Zhang, Siwei and Ma, Qianli and Zhang, Yan and Qian, Zhiyin and Kwon, Taein and Pollefeys, Marc and Bogo, Federica and Tang, Siyu},
   booktitle = {European conference on computer vision (ECCV)},
   month = oct,
   year = {2022}
}
```


# License
* Third-party software employs their respective license. Here are some examples:
    * Code/model/data relevant to the SMPL-X body model follows its own license.
    * Code/model/data relevant to the EgoBody dataset follows its own license.
    * Blender and its SMPL-X add-on employ their respective license.

* The rests employ the **Apache 2.0 license**. When using the code, please cite our work as above.




# Installation

### Environment
* **Tested OS:** Ubuntu 20.04, Windows 10, Windows 11
* **Packages:** Primary software packages are listed below. See `requirements.txt` for details.
    * Python >= 3.8
    * [PyTorch](https://pytorch.org) >= 1.8
    * [Open3D](http://www.open3d.org) == 0.13.0
    * torchgeometry == 0.1.2
    * [smplx](https://smpl-x.is.tue.mpg.de)
    * [vposer](https://github.com/nghorbani/human_body_prior): checkpoint version 1.0


### Datasets
* [**EgoBody**](https://amass.is.tue.mpg.de): We trained the motion prior by using the body model of camera-wearer in EgoBody dataset, and tested the motion prior with the body model of the interactee in EgoBody. 

### Paths
* To run the code properly it is important to set the paths of data, body model, and others. See `exp_GAMMAPrimitive/utils/config_env.py` for more details.

### Canonicalization
* required for training and visualizing motion primitives.
* `exp_GAMMAPrimitive/optimize_latent_var.py` to performs canonicalization for test data: the interactee. 
* `N_MPS` is the number of primitives in each sub-sequence. We set to 1 and 3, respectively. So we prepare **two** processed datasets with 10 or 30 frame per sequence.
* If you want to add any other markers in GRAB Marker set, edit the argument when calling get_grab_marker(). 

### Models
* [**SSM2 67** marker placement](https://drive.google.com/file/d/1ozQuVjXoDLiZ3YGV-7RpauJlunPfcx_d/view?usp=sharing)
* [**Pre-trained Checkpoints And cfgs**](https://drive.google.com/drive/folders/15IVBvXWmSvRlsspgomtLiwqj3iOZClSX?usp=sharing): put the checkpoints folders into `results/exp_GAMMAPrimitive/`. 
* [**GRAB** markerset](https://github.com/otaheri/GrabNet)

### Visualizating Motion in Blender
* Blender 2.93 or above.
* Install the [SMPL-X add-on](https://www.youtube.com/watch?v=DY2k29Jef94)

# Motion generation and reconsturction

### Every training and test step without 2D reconsturction on PV image, refers to GAMMA: https://github.com/yz-cnsdqz/GAMMA-release

### Reconstruct 2D image with body model on it:
* The best-fit latent variable is found by decreasing the mse loss between the ground truth openpose joints position and the corresponding joints on the predicted body model after projection to same camera plane.
```
python exp_GAMMAPrimitive/recover.py --cfg MPVAECombo_2frame_female_v10_grab_openpose
```
### Visualize the result:
* run exp_GAMMAPrimitive/vis_GAMMAprimitive.py, the 3D predicted motion and the 2D image with reconsturcted body model will be produced. The output path is printed out. 
* run python exp_GAMMAPrimitive/visualize_to_video.py to turn image outputs of vis_GAMMAprimitive.py to mp4 video.
# Result example:

### Comparison:
* Before the best-fit latent variable selected: 







https://user-images.githubusercontent.com/38854438/193464285-5c24f942-08c2-480e-9027-e327afbadc23.mov





* With the best-fit latent variable:





https://user-images.githubusercontent.com/38854438/193464186-42ba9bdb-8e6d-4aed-ab9e-22c73f413475.mov







