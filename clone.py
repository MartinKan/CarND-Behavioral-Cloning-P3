import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D

# Import driving data
lines = []
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# Divide the driving data into training samples (80%) and validation samples (20%)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Generator function to pipe the driving data samples into the model using smaller batch sizes
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			measurements = []
			for batch_sample in batch_samples:
				for i in range(3):  # Import the left, center and right camera views
					image_path = batch_sample[i]
					image_path = image_path.split('\\')[-1]
					image_path = 'IMG/' + image_path
					image = cv2.imread(image_path) # cv2.imread converts the image to the BGR color space
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # So we convert the image back to the RGB color space
					images.append(image)
					measurement = float(line[3])
					if i == 1:
						measurement += 0.2 # Right correct the steering angle of the left camera views 
					elif i == 2:
						measurement -= 0.2 # Left correct the steering angle of the right camera views
					measurements.append(measurement)

			# Augment the training data by flipping all of the images vertically
			augmented_images, augmented_measurements = [], []
			for image, measurement in zip(images, measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				augmented_images.append(cv2.flip(image,1))
				augmented_measurements.append(measurement*(-1.0))

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			yield sklearn.utils.shuffle(X_train, y_train)


# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Implements the NVIDIA deep learning architecture
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2,2), border_mode='same'))
model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2,2), border_mode='same'))
model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2,2), border_mode='same'))
model.add(Convolution2D(64, 5, 5, activation='elu', border_mode='same'))
model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=5)

model.save('model.h5')

