# Traffic Sign Classifier
## Overview
This project uses deep neural networks and convolutional neural networks to classify traffic signs. Specifically, a model is trained to classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. Here is the link to my [project code](Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration
I used the pandas library to calculate summary statistics of the traffic signs data set:

- The size of training set = 34799
- The size of validation set = 4410
- The size of test set = 12630
- The shape of a traffic sign image = 32x32x3
- The number of classes/labels = 43

The distribution of the training, validation and testing datasets is shown below:

<img src="img/distribution.JPG" width="75%" height="75%">

The histograms indicate that the class distribution is very similar between each dataset. It also allows to understand which signs are the most/least common in the datasets. To quantify this, let's count the 5 most and least common classes in the training dataset:

**5 Most common signs:**

|   |Sign name              |Id  |  Number of samples|
|---|-----------------------|:--:|:-----------------:|
|1. |Speed limit (50km/h)   | 2  |2010               |
|2. |Speed limit (30km/h)   | 1  |1980               |
|3. |Yield                  | 13 |1920               |
|4. |Priority road          | 12 |1890               |
|5. |Keep right             | 38 |1860               |

Few randomly selected images representing each class in the 5 most common cases:

<img src="img/5_Most_common_signs.JPG" width="75%" height="75%">

**5 Least common signs:**

|   |Sign name              |Id  |  Number of samples|
|---|-----------------------|:--:|:-----------------:|
|1. |End of all speed and passing limits   | 32  |210               |
|2. |Pedestrians   | 27  |210               |
|3. |Go straight or left                   | 37 |180               |
|4. |Dangerous curve to the left          | 19 |180               |
|5. |Speed limit (20km/h)             | 0 |180               |

Few randomly selected images representing each class in the 5 least common cases:

<img src="img/5_Least_common_signs.JPG" width="75%" height="75%">

### Data Preprocessing
Before using the images to train the neural network, some image preprocessing is applied to the training dataset. 
#### Convert RGB images to Grayscale
The initial test images are stored as RGB colorspace arrays. This is a 3 channel representation where the values for each channel (Red, Green, Blue) can go from 0 (dark) to 255 (bright).
In this step transformation from RGB space is applied to convert the image into Grayscale. 

The conversion from a RGB image to gray is done with:

```python
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```

The reason to convert the image to grayscale is to reduce # color channels and therefore make the neural network training faster. 
The assumption in this case is that color is not relevant for the traffic sign recognizer. One example where color would be of utmost importance would be traffic light recognizer and in that case RGB to GRAY scale conversion must not be applied!

#### Image Normalization
This processing changes the range of pixel intensity values. There are many algorithms to normalize an image. 
In this case the simplest way to normalize each image was used by dividing the pixel value by 255 and so normalizing each image to have value range between 0-1. Normalization helps in neural network training phase as it ensures the value range is similar between all features and no dominating features are present in the dataset.   

Here is few examples of traffic sign images before

<img src="img/original.JPG" width="75%" height="75%">

and after preprocessing:

<img src="img/preprocessed.JPG" width="75%" height="75%">

### Model Architecture
The infamous LeNet convolutional neural network structure (Yann LeCun et al. in 1998) was used as a reference for model structure.

As a first step I modified the LeNet model by simply adapting the last output layer from 10 to 43 outputs which coresponds to the number of classes of the traffic sign dataset. Then I increased the number of `EPOCHS` to 70 while keeping `BATCH_SIZE` = 128 and `rate` = 1e-3. The reason to increase the number of EPOCHS is due to the fact that the optimizer is gradient-based which means that in order to get to the minimum of the loss function, we need to perform multiple steps. The more steps we perform the more we get to the minimum (we ussualy get to some local minimum rather than global minimum). While the accuracy increased to about 91%, it also started to 'oscilate' meaning the accuracy sometimes increases then went down and again increased. I was trying to figure out what is the cause of those 'oscilations' and why the accuracy did not reach higher values. 

I susspected two things to be the cause of this issue:
1. Learning rate was to high
2. Number of units in the fully connected layer was not sufficient for the model to learn the data accurately

So I decided to futher change the neural network architecture as follows:
- decrease learning rate to 8e-4 (although I also tried to ge as low as 5e-5 which proved to be to small and therefore the training would take to long to complete)
- increase the number of units in the 1. fully conected layer from 120 to 200 units
- add 3 dropout layers with 60% keep probability to make sure the neural network doesn't overfit the training data

Here is the final model architecture:
|Layer|Description|
|---|---|
|Input| 32x32x1 image|
|Convolution 5x5| 1x1 stride, VALID padding, output 28x28x6 |
|relu| activation function|
|Max Pooling|2x2 kernel, 2x2 stride, VALID padding, output 14x14x6|
|Convolution 5x5| 1x1 stride, VALID padding, output 10x10x16 |
|relu| activation function|
|Max Pooling|2x2 kernel, 2x2 stride, VALID padding, output 5x5x16|
|Dropout|keep probability 60%|
|Fully connected|input 400, output 200|
|relu| activation function|
|Dropout|keep probability 60%|
|Fully connected|input 200, output 82|
|relu| activation function|
|Dropout|keep probability 60%|
|Fully connected|input 82, output 43|

### Train and Test NN Model
To train the model, I used Adam Algorithm which is an extension of stochastic gradient descent. Whereas the Stochastic gradient descent maintains a single learning rate for all weight updates and the learning rate does not change during training, Adam computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

To make sure the training is not skewed by the order of images in the training set, the implemented training pipeline randomly shuffles the training set each time before next epoch start.

Some hyperparameters I chose for the nn training: 

- `EPOCHS` = 70
- `BATCH_SIZE` = 256
- `rate` = 8e-4

My final model results were:

- training set accuracy: 99%
- validation set accuracy: 96%
- test set accuracy: 95%

### Test the Model on New Images

Here are five German traffic signs that I found on the web:

<img src="img/new_images_original.JPG" width="75%" height="75%">

Altough the images I found are of good quality, they initially were a part of a bigger image with a landscape and one image included two traffic signs. The classification of those initial images end up to be around 60%. This is due to the fact that I resized each image to 32x32px which largely decreased the quality, skewed the images and make the traffic sign almost unrecognizable. So I decided to manualy crop each image around the traffic sign to make sure the sign image doesn't get to distorted and those are the images shown above.

From the visual inspection these images shouldn't be difficult to classify as you can see below.

I applied the same preprocessing to these images as to the training dataset. Running the images through the traffic sign classifier results in prediction accuracy of 100%.

|Label|Prediction|
|---|---|
|Yield|Yield|
|Speed limit (50km/h)|Speed limit (50km/h)|
|Priority road|Priority road|
|Stop|Stop|
|Double curve|Double curve|

A direct prediction accuracy comparison with the training set should not be made since the size of both datasets is significantly different and the results might be missleading. Much better way would be to test the prediction accuracy on larger dataset (100-1000 images) than on just 5 images. 

For each of the 5 images, I printed out the model's softmax values to show the certainty of the model's predictions.
The function `tf.nn.top_k` was used to select the top 5 class probabilities predicted by the traffic sign classifier.

<img src="img/new_images_top_5_classes.JPG" width="75%" height="75%">

The highest softmax value of each image is accurately predicting the correct traffic sign class. The softmax value of the last image is a little bit lower. This might be due to the preprocessing operation that resized the original image to 32x32px which caused the image to be slightly deformed.
The classifier is very certain of its predicitions not making any significant prediction errors. 
