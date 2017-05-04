**Traffic Sign Recognition** 

Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization1.png "Visualization 1"
[image1_1]: ./examples/visualization2.PNG "Visualization 2"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image2_2]: ./examples/gray2.png  "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image3_3]: ./examples/adddata.png "Random Noise"
[image4]: ./testImages/img1.png "Traffic Sign 1"
[image5]: ./testImages/img2.png "Traffic Sign 2"
[image6]: ./testImages/img3.png "Traffic Sign 3"
[image7]: ./testImages/img4.png "Traffic Sign 4"
[image8]: ./testImages/img5.png "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points.

Here is a link to my [project code](https://github.com/asimay/SDC-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set: 

The code for this step is contained in the first code cell of the IPython notebook.  

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 40000（original is 34799) 
* The size of test set is 12630 
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. The exploratory visualization of the dataset:

The code for this step is contained in the third and fourth code cell of the IPython notebook.  

Here is random input images and corresponding labels visualization:

![random input images and corresponding labels][image1_1]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![bar_chart:Distribution of labels in training data][image1]

### Design and Test a Model Architecture

#### 1. images Pre-processing techniques:
such as converting to grayscale, normalization, etc.

The code for this step is contained in step 2 first code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the result is color irrelant.

Here is an example of a traffic sign image before and after grayscaling.

![gray_scale][image2_2]

As a last step, I normalized the image data because this can reduce the impact of too big weight image.

#### 2. set up training, validation and testing data:

Extract the training, validation, testing data from the zip file, and load into corresponding data set.

The code is contained in the first code cell of the IPython notebook.  

My final training set had **40000** number of images. 

My validation set and test set had **8000 (original is 4410)** and **12630** number of images.

The code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because to avoid under-fitting. To add more data to the the data set, I used the following techniques : 

Here is an example of an original image and an augmented image:

![original image and an augmented image][image3_3]

The difference between the original data set and the augmented data set is : augmented data add some Gaussian noise in original image. 


#### 3. final model architecture (including model type, layers, layer sizes, connectivity, etc.) looks like : 

The code for my final model is located in the Model Architecture cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	   | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Flatten  	         | Input = 5x5x64. Output = 1600			|
| Fully connected	1	 | Input = 1600. Output = 1000			  |
| Fully connected	2	 | Input = 1000. Output = 500			   |
| Fully connected	3  | Input = 500. Output = 43			     |
|						|												|
|						|												|
 


#### 4. trained model:

The code for training the model is located in the cell of "Train, Validate and Test the Model" of the ipython notebook. 

To train the model, I used "AdamOptimizer" and "evaluate" function to optimize and evaluate the result.

batch size is :128

number of epochs: 20, but it can stop till it check the validation accuracy meet the requirement.

hyperparameters:

    mu = 0
    sigma = 0.1
    learning_rate = 0.001
    drop_out = 0.3
    max_acc = 0.0
    THRESHOLD = 0.95

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in cell of "Train, Validate and Test the Model" of the Ipython notebook.

After 5 Epochs My final model results were:
* training set accuracy of 87.8%
* validation set accuracy of 95.2% 
* test set accuracy of 93.8%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

  ==> LeNet-5 architecture is the first choose.

* What were some problems with the initial architecture?

  ==> Filter Depth is not enough.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  ==> Add dropout layer.

* Which parameters were tuned? How were they adjusted and why?

  ==> Filter Depth is tuned.


If a well known architecture was chosen:
* What architecture was chosen?

  ==> LeNet-5 architecture

* Why did you believe it would be relevant to the traffic sign application?

  ==> it has similar characteristics with MNIST.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

  ==> We need to test the Module to prove the model is good, and test result is 93.8%.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
