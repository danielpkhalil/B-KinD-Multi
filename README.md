# B-KinD-Multi: Learning Keypoints for Multi-Agent Behavior Analysis using Self-Supervision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19cxahDPwxu_qbOLKocUbDm6Rm9gjmFk4?usp=sharing)

<p align="center"><img src="docs/static/images/cover.png" width="80%" alt="" /></p> 


Implementation from the paper:
>Daniel Khalil, Christina Liu, Pietro Perona, Jennifer Sun, Markus Marks, [Learning Keypoints for Multi-Agent Behavior Analysis using Self-Supervision](https://arxiv.org/pdf/2112.05121.pdf). In Conference on Computer Vision and Pattern Recognition (CVPR), 2022

B-KinD-multi discovers keypoints without the need for bounding box annotations or manual keypoint, and works on a range of organisms and any number of agents.

<p align="center"><img src="figs/bkind_gif.gif" width="60%" alt="" /></p> 

Our code currently supports running keypoint discovery on your own videos, where there's relatively stationary background and no significant occlusion. 

# Quick Start
To set up the environment, here are all the install commands we used (you may need to change the cudatoolkit version depending on your GPU):

```conda create --name env python=3.10.12```

```conda activate env```

```git clone https://github.com/hkchengrex/Grounded-Segment-Anything```

```conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia```

```wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run```

```sudo apt install gcc```

```sudo sh cuda_11.7.0_515.43.04_linux.run```

```cd Grounded-Segment-Anything/```

```export BUILD_WITH_CUDA=True```

```export AM_I_DOCKER=False```

```export CUDA_HOME=/usr/local/cuda-11.7```

```cd Grounded-Segment-Anything/```

```export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}```

```export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}```

```export PATH=$PATH:/usr/local/bin/aws```

```export PATH=$PATH:[MYHIP]/bin```

```sudo apt-get install g++```

```pip install -e GroundingDINO (may have to --force-reinstall if it doesn't install first time without errors)```

```pip install -q -e segment_anything```

```git clone https://github.com/danielpkhalil/Tracking-Anything-with-DEVA```

```cd Tracking-Anything-with-DEVA/```

```pip install -q -e .```

```bash scripts/download_models.sh```


```conda install pytorch torchvision cudatoolkit=11.3 -c pytorch```

```conda install -c anaconda pyyaml```

```conda install -c anaconda h5py```

```pip install piq```

```pip install opencv-python```

```pip install seaborn```

```pip install imageio```

## CalMS21
1. Download CalMS21 dataset: https://data.caltech.edu/records/1991
   - For training keypoint discovery (contains videos): task1_videos_seq.zip
   - For training behavior classification (contains behavior annotations for evaluation): task1_classic_classification.zip
2. Extract frames from video
   -  Use the code provided by CalMS21 in seq_image_extraction.zip to extract images for all the downloaded seq files.
   - There should be one image directory for the train split, and one image directory for the test split.
     Within each directory, there should then be directories of images corresponding to each video.
3. Run command
```
python train_video.py --config config/CalMS21.yaml 
```
This will take 1~2 days on a single GPU, the keypoints converge around epoch 10.

### Behavior classification
1. One you are done training keypoint discovery model, evaluate it on the CalMS21 task 1 data on the default split.
   The classification code is provided by [1]: https://gitlab.aicrowd.com/aicrowd/research/mab-e/mab-e-baselines
2. To evaluate using our keypoints, first extract keypoints for all images in CalMS21 task 1:
```
python extract_features.py --train_dir [images from train split CalMS21] --test_dir [images from test split CalMS21]
 --resume [checkpoint path] --output_dir [output directory to store keypoints]
```
3. Convert the extracted keypoints to the npy format for the CalMS21 baseline code:
```
python convert_discovered_keypoints_to_classifier.py --input_train_file [calms21_task1_train.json from step 1b) above] --input_test_file [calms21_task1_test.json from step 1b) above] --keypoint_dir_train [keypoints_confidence/train extracted in previous step]  --keypoint_dir_test [keypoints_confidence/test extracted in previous step]
```
The data is automatically saved into the data/ directory.

4. Use the saved npy files in data in the CalMS21 classifier code (train_task1.py). Note that the default code from [1] only handles input keypoints. We provide a modified version that reads in keypoints, confidence and covariance in the classifier/ directory.
```
python classifier/train_task1.py **
```
** Need to set train_data_path and test_data_path inside the train_task1.py file to the files generated from the previous step.


## Human 3.6M
1. Download Human 3.6M dataset: http://vision.imar.ro/human3.6m/description.php (Ask for permission to authors)
2. Extract frames from videos
   - Our implementation uses this code (https://github.com/anibali/h36m-fetch) for frame extraction
   - Dataloader for h36m (dataloader/h36m_dataset.py) has to be updated if you extract frames using a different code.
3. Run command
```
python train_video.py --config config/H36M.yaml
```

### Pose regression on Simplified Human 3.6M
1. Once you are done with training keypoint discovery model, evaluate pose regression task on Simplified Human 3.6M dataset
2. Simplified Human 3.6M Dataset is publicly available here from [2]: http://fy.z-yt.net/files.ytzhang.net/lmdis-rep/release-v1/human3.6m/human_images.tar.gz
3. Run command
```
python test_simplehuman36m.py [path_to_simplified_h36m_dataset**] --checkpoint [path_to_model_directory] --batch-size [N] --gpu 0 --nkpts [K] --resume [path_to_model_file]
```
*Note that [path_to_h36m_dataset] should end with 'processed' directory
**[path_to_simplified_h36m_dataset] should end with 'human_images' directory

Regression results may vary since our method does not use any keypoint label as a supervision signal while training the keypoint discovery model.


## Your own dataset
Please follow the instructions below if you would like to train B-KinD on your own video dataset! Note that the code currently does *not* support: tracking for multiple agents with similar appearance, videos with a lot of background motion, and/or videos with a lot of occlusion or self-occlusion. 

Note that one important consideration is the ```frame_gap``` parameter in the config directory, you want to pick a gap where the agent has moved a bit, but not too much (usually this is under 1 second, we used frame gap corresponding to 0.2s in our work). Also adjust ```nkpts``` to the number of parts you want to track. 

1. Extract frames and put all the frames in the "data/custom_dataset" directory. There are many ways to extract frames - for example, using ffmpeg:
```
ffmpeg -i [input_video_name] [output_directory]/images_%08d.png
```
2. Directory structure should follow the format below:
```
data/custom_dataset
    |---train
        |---video1
            |---image0.png
            |---image1.png
            |---    .
            |---    .
        |---video2
        |---   .
        |---   .
    |--val
        |---videoN
```
*Image file names should be in an ascending order for sampling pair of images sequentially

3. Set up the configuration of your data in ``config/custom_dataset.yaml``
4. Run command
```
python train_multi_agents.py --config config/custom_dataset.yaml
```
Use the ```--visualize``` flag to visualize examples during training. This can be helpful to see when the keypoints have converged. 

5. To extract additional features from the discovered heatmap, run command
```
python extract_features.py --train_dir [images from train split] --test_dir [images from test split]
 --resume [checkpoint path] --output_dir [output directory to store keypoints] --imsize [training image size] --nkpts [number of discovered keypoints]
```
6. To run inference on new frames, put images and a directory named "sample_images" and run command

``` python multi_agent_inference.py ```

## License

Please refer to our paper for details and consider citing it if you find the code useful:
```
@article{bkind2021,
  title={Self-Supervised Keypoint Discovery in Behavioral Videos},
  author={Sun, Jennifer J and Ryou, Serim and Goldshmid, Roni and Weissbourd, Brandon and Dabiri, John and Anderson, David J and Kennedy, Ann and Yue, Yisong and Perona, Pietro},
  journal={arXiv preprint arXiv:2112.05121},
  year={2021}
}
```

Our code is available under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

[1] Sun et al., The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions. NeurIPS 2021.

[2] Zhang et al., Unsupervised discovery of object landmarks as structural representations. CVPR 2018.
