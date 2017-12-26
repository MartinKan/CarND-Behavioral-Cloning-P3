## Udacity - Car Nanodegree - Behavioral Cloning Project

Overview
---

This repository contains the following files that I will be submitting for the behavioral cloning project:

1.  [model.py](https://github.com/MartinKan/CarND-Behavioral-Cloning-P3/blob/master/model.py)
2.  [model.zip](https://github.com/MartinKan/CarND-Behavioral-Cloning-P3/blob/master/model.zip)
3.  [drive.py](https://github.com/MartinKan/CarND-Behavioral-Cloning-P3/blob/master/drive.py)
4.  [writeup_report.md](https://github.com/MartinKan/CarND-Behavioral-Cloning-P3/blob/master/writeup_report.md)
5.  [video.mp4](https://github.com/MartinKan/CarND-Behavioral-Cloning-P3/blob/master/video.mp4)

The first file contains the code of the model that I have written for the project.  

The second file is the compressed file of model.h5 (compression is required to reduce the file size to below 25MB), which is the output of the first file.  The model.h5 file is used in conjunction with drive.py to autonomously drive the car around track 1 in the simulator.

The third file is the script that was provided to us in the project files (albeit with some minor amendments by me) that is used to run model.h5.

The fourth file is a written report for my project that addresses each of the rubric points for the project.

The fifth file is a video capture of the car making 2 laps around track 1 in autonomous mode when model.h5 was running in the background.

My code (first file) contains a deep learning network that can be used to train a car to run autonomously around track 1 in the car simulation program that was provided with the project.  A further explanation on how it works can be found in the write up.
