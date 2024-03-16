# Mirror-Aware Neural Humans 🏃🏻🪞

### [Paper](https://arxiv.org/abs/2309.04750) | [Supplementary](https://danielajisafe.github.io/mirror-aware-neural-humans/docs/Supp.pdf) | [Website](https://danielajisafe.github.io/mirror-aware-neural-humans/) 
![](imgs/front.png)
>**Mirror-Aware Neural Humans**\
>[Daniel Ajisafe](https://danielajisafe.github.io/), [James Tang](https://www.linkedin.com/in/james-tang-279332196/?originalSubdomain=ca), [Shih-Yang Su](https://lemonatsu.github.io/), [Bastian Wandt](https://bastianwandt.de/), and [Helge Rhodin](http://helge.rhodin.de/)\
>The 11th International Conference on 3D Vision (3DV 2024)

#### Updates
- Feb 6, 2023: Codebase setup.
- Feb 14, 2023: Stage 1 and 2 code released.

## Setup
```
git clone git@github.com:danielajisafe/Mirror-Aware-Neural-Humans.git
cd Mirror-Aware-Neural-Humans
```
The conda environment provides support for packages required in all three stages (1,2,3).
```
conda create -n mirror-aware-human python=3.8
conda activate mirror-aware-human

# install pytorch for your corresponding CUDA environments
pip install torch==2.0.0 # (recommended)

# install pytorch3d: note that doing `pip install pytorch3d` directly may install an older version with bugs.
# be sure that you specify the version that matches your CUDA environment if the command below does not work for you. See: https://github.com/facebookresearch/pytorch3d
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt190/download.html

# install other dependencies
pip install -r requirements.txt
```

## Training

#### Stage 1 

We reconstruct camera intrinsics and ground normal calibrations using 2D keypoints as anchors, based on our [CasCalib implementation](https://github.com/tangytoby/CasCalib/tree/main). You can find the focal and ground normal reconstruction ```focal_ground_normal``` in this [Google drive](https://drive.google.com/drive/folders/1hTA1BKX63UeulJUixS1vo8hLqsbpX2AA?usp=sharing) for ```camera 3 (eval)``` and ```subject 1 (non-eval)```. Please place all items in the drive into the appropriate directory below. 

```
cd dataset && mkdir eval non_eval visualai visualai/images
```

Your directory tree should look like this:
```
Mirror-Aware-Neural-Humans
├── core_mirror
└── dataset
   ├── intermediate.md 
   └── visualai
      └── images
   ├── eval
   └── non_eval
└── DANBO-pytorch
   └── ...
├── extras
├── vis
   └── vis.ipynb
├── README.md
└── requirements.txt
```

<!-- ```
Mirror-Aware-Neural-Humans
├── core_mirror
├── dataset
   ├── intermediate.md 
   ├── zju-m-seq1
      ├── images
      └── ...
   ├── visualai
      └── images
   ├── eval
   ├── non_eval
   └──calibration
      ├── Cam3
      └── Subj3
├── outputs
├── smpl_files
├── models
├── README.md
└── environment.yml
└── requirements.txt
``` -->

To run stage 1, and prepare for stage 2 and 3, please follow these [intermediate steps](https://github.com/danielajisafe/Mirror-Aware-Neural-Humans/blob/main/dataset/intermediate.md) carefully.


#### Stage 2

This stage follows the classical multi-view optimization going from 2D to 3D but with a single camera. We start from a template rest pose, which comes from the first frame in the [H3.6M dataset](http://vision.imar.ro/human3.6m/description.php). Please set the flag ```--h36m_data_dir``` in the command below to where your H3.6M data is located. 

We are not allowed to share the pre-processed template pose due to license terms, please reach out to ```dajisafe[at]cs.ubc.ca``` with the subject title ```Mirror-Aware-Human Template Pose``` if you need access, and set ```--h36m_data_dir``` to ```None```.

You can reconstruct the 3D pose with the following command:
```
# on eval sequence
python3 -m core_mirror.optimize_general --project_dir '' --h36m_data_dir /path/to/h36m/dir --eval_data --view 3 --useGTfocal --use_mapper --alphapose --start_zero --disable_rot --skel_type "alpha" --loc_smooth_loss --orient_smooth_loss --feet_loss --body15 --print_eval --infostamp user --iterations 2000

# on non-eval sequence
python3 -m core_mirror.optimize_general --project_dir '' --h36m_data_dir /path/to/h36m/dir --rec_data --view 0 --use_mapper --alphapose --start_zero --disable_rot --skel_type 'alpha' --loc_smooth_loss --orient_smooth_loss --feet_loss --opt_k --seq_name Subj3 --infostamp user --iterations 2000
```

Here, 
- ```--rec_data``` specifies non-eval data, 
- ```--view``` specifies the camera ID for eval data, and ```0``` for non-eval, 
- ```--opt_k``` refines the estimated focal length (from stage 1) in stage 2, 
- ```--use_mapper``` converts between different skeleton configurations, 
- ```--body15``` uses the common 15 joints between alphapose and mirror skeleton, 
- ```--start_zero``` sets starting rotations to ```0``` degree,
- ```--disable_rot``` disables optimization for feet and face rotations, and
- ```--loc_smooth_loss```, ```--orient_smooth_loss```, and ```--feet_loss``` enforces additional constraints on the joint positions, joint orientations, and feet-to-ground distance respectively.

The reconstruction results can be found in `outputs/`.
	
The 3D outputs can also be visualized in the jupyter notebook ```vis/vis.ipynb```.

The results from [stage 1]() and [stage 2]() is used to prepare data for training the neural model (stage 3). We provide the pre-processed data in ```.h5``` format for two characters ```camera 3 (eval)``` and ```subject 1 (non-eval)```. Please see [drive](https://drive.google.com/drive/folders/1hTA1BKX63UeulJUixS1vo8hLqsbpX2AA?usp=sharing) and kindly cite the [data source](https://github.com/zju3dv/Mirrored-Human/) for the eval set appropriately.  Move the ```data``` folder under "body_h5" from google drive to the ```DANBO-pytorch/``` directory.

#### Stage 3

Coming soon ...


## Citation
if the code is helpful to your research, please consider citing and giving us a ⭐ :
```
@article{ajisafe2023mirror,
title={Mirror-Aware Neural Humans},
author={Ajisafe, Daniel and Tang, James and Su, Shih-Yang and Wandt, Bastian and Rhodin, Helge},
journal={arXiv preprint arXiv:2309.04750},
year={2023}
}
```
```
@misc{CasCalib,a@1
title={CasCalib: Cascaded Calibration for Motion Capture from Sparse Unsynchronized Cameras},
author={Tang, James and Suri, Shashwat and Ajisafe, Daniel and and Wandt, Bastian and Rhodin, Helge},
note ={Technical report},
year={2023}
}
```

## Acknowledgements
Our code is built mainly on the generous open-source efforts of prior works, including [A-NeRF](https://github.com/LemonATsu/A-NeRF), [DANBO](https://github.com/LemonATsu/DANBO-pytorch), and [Mirror-Human](https://github.com/zju3dv/Mirrored-Human).
