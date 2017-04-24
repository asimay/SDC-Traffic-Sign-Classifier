**Traffic Sign Recognition**

*Write up for traffic sign Classifier*

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

[image1]: ./examples/prediction1.png "Correct Predicted Double Curve"
[image2]: ./examples/prediction2.png "Correct Predicted Keep Left"
[image3]: ./examples/prediction3.png "Wrong Predicted Roundabout mandatory"
[image4]: ./examples/prediction4.png "Correct Predicted Roundabout mandatory"
[image5]: ./examples/prediction5.png "Wrong Predicted Turn left Ahead"
[image6]: ./examples/prediction6.png "Wrong Predicted Childern Crossing"
[image7]: ./examples/prediction7.png "Wrong Predictied 20km/h speed limit"
[image8]: ./examples/prediction8.png "Correct Predictied 20km/h speed limit"
[image9]: ./examples/preprocess.png  "Preprocessed Image"
[image10]: ./examples/sign_classes.png "sign-classes"
[image11]: ./examples/test-images.png "test-images"
[image12]: ./examples/weights.png "Network weight 1"
[image13]: ./examples/weights-2.png "Network weight 2"
[image14]: ./examples/weights-3.png "Network weight 3"
[image15]: ./examples/weights-4.png "Network weight 4"
[image16]: ./examples/image.png "Training Image"


## Rubric Points
*I have impletemented Traffic Sign Classifier using the LeNet Neural Network Architecture and Apply preprocessing on the dataset. Also tested the network on Validation Set / Test Set provided by [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and used some of images downloaded from web to predit the sign. *

---
*Step 1: Load the Data:*
* Number of training examples = 34799
* Number of testing examples = 12630
* Number of Validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

*Step 2: Analyse the Data:*
I visualize the classes count and it seems that the distribution of the dataset is not balanced, there are classes contain many examples but some classes had very few examples.
![alt text][image10]

*Step 3: Visualize the image*
![alt text][image16]

*Step 4: Preprocess the image*
* Convert RGB to Gray Scale image.
* Normalize the Gray scale image to value between 0 and 1.
* Applied Histogram Equilization
![alt text][image9]

*Step 5: Define hyperparameters*
* Learning Rate set to 0.001
 REASON: I have tried different learning rate as well. As I increase learning rate then network trained on training set very well but have fluctuation on the validation set, On decreasing learning rate network tends to ramp up the accuracy very slow but after 200 epochs network accuracy on validation set doesn't increase. Hence I have sattled on this value.

* Number of Epochs set to 100
 REASON: With above learning rate network reaches to 97% accuracy on validation set. I have trained the netwrok once for 200 epochs and see that the validation accuracy on getting increased after 70 epochs. Hence i have kept the number of epoch to 100 and then make use of THRESHOLD value for validation accuracy to 96%. I have choosen 96% because as i trained my network more it will overfit with training data as training accuracy reaches to 99.7%.

* Batch size set to 128
REASON: I have Nvidia TitanX GPU so this batch size works fine.

* Regularization - Dropout
REASON: As My network easliy got 100% accuracy on training set after 4-5 epochs, I have choosen to apply regularization so that network will not overfit on training set. Hence apply two Dropout layer with probability of keeping neuron set 0.3. (Also this is one of the suggestion from review)

*Step 6: Define Network Architecture*
* Used LeNet Architecture same as what used in LeNet Lab in Class
All weights initialized using Gaussian distribution with mean 0 and standard deviation 0.1
My final model consisted of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 Normalized Gray image                 |
| Convolution 5x5x6     | 1x1 stride, valid padding, outputs 28x28x6    |
| Max pooling           | 2x2 stride, outputs 14x14x6                   |
| RELU                  |                                               |
| Convolution 5x5x16    | 1x1 stride, valid padding, outputs 10x10x16   |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| RELU                  |                                               |
| Fully connected       | Input 400 Outputs 120                         |
| RELU                  |                                               |
| Dropout               | Keep prob 0.3                                 |
| Fully connected       | Input 120 Outputs 84                          |
| RELU                  |                                               |
| Dropout               | Keep prob 0.3                                 |
| Fully connected       | Input 84 Outputs 43                           |
| Softmax               | Output probabilities of 43 classes            |
|                       |                                               |


*Step 7: Add Visulization feature of the weights getting trained*
after few steps of training the weights changes and learn to detect the edges / circle etc.

After few steps of training, weight of second convolution layer changes as following images:
![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

*Step 8: Define the training / validation formula*
* Use softmax on cross entropy
* Used Adam Optimizer for Backpropogation

*Step 9: Training the Model*
* On Validation data Accuracy reaches to 93% after 9 epochs.
* Maximum validation accuracy reached of 97.4% after 61 epochs

*Step 10: Evaluate the model on Test Set*
* On Test data set Accuracy reaches to 94.8%.

*Step 11: Downloaded the German Sign Images from Web and preprocessed*
* I have used 33 images downlaoded from the web to evaluate the performace.
* On these images model gives accuracy of 78.8%.

Preprocessed Test Images:
![alt text][image11]

*Step 12: Analyze the images*
I have found out that on the most of the images downloaded from web classified correctly correctly. But some image not getting classified correctly. Few reasons:
* Training set doesn't include many example of worngly classified image (e.g. Speed Limit (20 Km/h), Roundabout mandatory,  etc.)
* Some Images are skewed and hence required dataset to be contain images which are skewed from there original version.
* Most of the wrong classified images as effect of sun light / Shades.

*Following are the example of classification on test images from web*
![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

Points to improve:
* Most important part is to balance the data set.
* Augment the dataset to provide more images using skewing / Translation / rotation.
* Try few more network.
