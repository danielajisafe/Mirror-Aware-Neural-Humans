### Intermediate (I) steps

##### Preparing the directory

I-Step A: 
- Download and extract the [mirror-eval dataset](https://github.com/zju3dv/Mirrored-Human/blob/main/doc/evaluation.md).
  
```
cd dataset/
data_url=https://www.dropbox.com/s/fjyhgbcbe8cpoa5/zju-m-seq1.zip?dl=0
wget ${data_url}
mv zju-m-seq1.zip* zju-m-seq1.zip 
unzip zju-m-seq1.zip -d zju-m-seq1
rm zju-m-seq1.zip
```

- Extract all frames from zju videos to work with images
  
```
mkdir zju-m-seq1/images zju-m-seq1/images/3
# extract all frames from video e.g camera 3
ffmpeg -i zju-m-seq1/videos/3.mp4 -start_number 0 'zju-m-seq1/images/3/%08d.jpg'
```

- Unzip all images from ```/visualai``` 
  
```
unzip visualai/images/frames.zip -d visualai/images
mv visualai/images/frames visualai/images/3
rm visualai/images/frames.zip
```

- Create calibration results folder 
  
```
cd ..
mkdir dataset/calibration/ dataset/calibration/Cam3 dataset/calibration/Subj3
```

### Stage 1

We reconstruct the ground plane using our single-view calibration method called ```CasCalib``` with 2D keypoint detections from [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) as anchors. The focal length can also be reconstructed if not available. By following the instructions there (step 1-to-3) (you need it😀) and running the following code in the CasCalib [repository](https://github.com/tangytoby/CasCalib/tree/main), 

```
# eval sequence
python run_single_view.py /path/to/dataset/zju-m-seq1/images/3/00000000.jpg /path/to/dataset/eval/Cam3_alphapose-results.json 1
# non-eval sequence
python run_single_view.py /path/to/dataset/visualai/images/3/00000000.jpg /path/to/dataset/non_eval/Subj3_alphapose-results.json 1
```
the reconstructed ground plane and focal length should be saved to a pickle and json file under ```outputs/single_view_[timestamp]``` within CasCalib. Please move these output files to the appropriate folders under ```dataset/calibration/``` within the mirror-aware-human repository.

I-Step B: 
- Be sure you are within the ```Mirror-Aware-Neural-Humans``` repository and ```mirror-aware-human``` environment is activated.
- Combine the zju annotation frame files to a single file per video. Camera 4 video is dropped as its frontal to the mirror.
- Convert alphapose detections to dcpose format as required in Stage 2.

```
python -m extras.preprocess
```

I-Step C: 
- Reconstruct initial 3D ankles from 2D detections via plane ray intersection. See results in ```/output``` folder.

```
python3 -m core_mirror.internal_calib.plane_ray_intersect --camera_id 3 --json_file "dataset/eval/Cam3_alphap2dcpose.json" --image_directory "dataset/zju-m-seq1/images/3/" --pickle_path 'dataset/calibration/Cam3/calibration.pickle' --output_dir outputs --infostamp user --skel_type "dcpose" --useGTfocal 

python3 -m core_mirror.internal_calib.plane_ray_intersect --camera_id 0 --json_file 'dataset/non_eval/Subj3_alphap2dcpose.json' --image_directory 'dataset/visualai/images/3' --skel_type 'dcpose' --seq_name 'Subj3' --pickle_path 'dataset/calibration/Subj3/calibration.pickle' --output_dir outputs --infostamp user"
```

Please proceed to [Stage 2](https://github.com/danielajisafe/Mirror-Aware-Neural-Humans?tab=readme-ov-file#stage-2) in the main repository.

