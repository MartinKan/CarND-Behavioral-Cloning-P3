# **Behavioral Cloning** 

## Project Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sample_images/Center_Driving_Example.jpg "Center Driving Example"
[image2]: ./sample_images/Recovery_Left_Side_1.jpg "Recovery Left Side 1"
[image3]: ./sample_images/Recovery_Left_Side_2.jpg "Recovery Left Side 2"
[image4]: ./sample_images/Recovery_Left_Side_3.jpg "Recovery Left Side 3"
[image5]: ./sample_images/Recovery_Right_Side_1.jpg "Recovery Right Side 1"
[image6]: ./sample_images/Recovery_Right_Side_2.jpg "Recovery Right Side 2"
[image7]: ./sample_images/Recovery_Right_Side_3.jpg "Recovery Right Side 3"
[image8]: ./sample_images/Augmented_1.jpg "Augmented 1"
[image9]: ./sample_images/Augmented_2.jpg "Augmented 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (clone.py lines 63-76) 

The model includes ELU layers to introduce nonlinearity (code lines 66-70 and 73-75). The data is normalized in the model using a Keras lambda layer (code line 64) and cropped using a Keras cropping layer (code line 65). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (clone.py lines 71). 

Augmented data was generated to help generalize the model (code lines 46-51).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 17-18 and 79). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 78).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, smooth cornering and recovery driving from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was not to reinvent the wheel but to implement something that was proven to work and adapt it for my own application.

My first step was to use a convolution neural network model similar to the one developed by the autonomous vehicle team in NVidia since it was both recommended by the course material and proven to work in the real world.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it contained a dropout layer.  

Then I created extra data to generalize my model and make it less specific to the test track.  This includes recording the driving data when driving in both directions on the test track and also augmenting the recorded data by flipping all of the images vertically thus doubling the amount of data available to train the model. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (especially during cornering). To improve the driving behavior in these cases, I recorded extra driving data.  I also found that by driving slower and decreasing the image quality of the simulator (both resolution and speed) I was able to collect more accurate driving data.  This helped in the process of training the model to drive more accurately.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 62-78) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Convolution 5x5 (L1) 	| 24 filters 	|
| ELU	|												|
| Convolution 5x5 (L2) 	| 36 filters 	|
| ELU	|												|
| Convolution 5x5 (L3) 	| 48 filters 	|
| ELU	|	
| Convolution 5x5 (L4) 	| 64 filters 	|
| ELU	|							
| Convolution 3x3 (L5) 	| 64 filters 	|
| ELU	& Dropout |										|			
| Flatten			  |    								|
| Fully connected (L6)	| Outputs 100  		|
| ELU	| 												|
| Fully connected (L7)	| Outputs 50  		|
| ELU	| 												|
| Fully connected (L8)	| Outputs 10  		|
| ELU	| 												|
| Fully connected (L9)	| Outputs 1   		|
| ELU	| 												|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on the track by itself. These images show what a recovery looks like starting from the right side of the road:

![alt text][image2]
![alt text][image3]
![alt text][image4]

These images show what a recovery looks like starting from the left side of the road:

![alt text][image5]
![alt text][image6]
![alt text][image7]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image8]
![alt text][image9]

After the collection process, I had X number of data points. I then preprocessed this data by normalizing ....

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
