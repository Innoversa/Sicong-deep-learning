## Introduction

The purpose of this project is to detect whether the person in an image is looking at the cell phone or not

## Pictures/Images

Please navigate to /my_pics/ directory\
op_pic_x.jpg == original pictures\
/output_pic/ directory == rendered pictures\
json_datas/ directory == key points for each pictures\
json_datas/ans_all_15_images.csv == key for the supervised training\
json_datas/input_all_15_images.csv == processed data


## Videos

go to /my_videos directory\
To view rendered videos, go to /output/result_vid_1.avi\
To view Json data vs keypoints vs time, please view /output/vid_1 directory\
my_videos/output/avi_x/avi_x_json_out.json == the json where each frame corresponds to the prediction label(confidence score)

## Usage

All the related code is stored in the sicong_4.py in the root directory
```bash
python3 sicong_4.py
```
## Demo
https://youtu.be/JDaMm2D2N4w

## Issues
Currently cannot display results in realtime (have to process the image then get answer from from bash

## Dependencies

```bash 
pip install matplotlib
pip install face_alignment
pip install sklearn
````