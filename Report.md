**Traffic Sign Recognition** 



### Data Set Summary & Exploration

#### 1. I loaded the provided training/validation/test files and extracted features and labels as NumPy arrays. I used the Python & NumPy functions "len", "shape", & "unique" to collect the following summary about the data. 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32X32X3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Visualization plots are included in the solution notebook.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* I converted images to grayscale by taking a weighted sum of the three color channels. The weights for the RGB channels are 0.299, 0.587, 0.114, respectively. Working on one channel instead of three improve the model's prediction accuracy. It prevents the model from overfitting the color artificats where the information in one channel is sufficient to classifiy the traffic signs effectively. Also, the training speed increases by utilizing less number of parameters in the first convolution layer.

* After converting the images to gray-scale, I normalized them by subtracting 128 from all pixels and dividing the result by 128. The value 128 is half the maximum number of pixel intensities 256.

The difference between the original data set and the augmented data set is around 10% in terms of prediction accuracy on the validation set.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray scale image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 400X120        									|
| Fully connected		| 120X84        									|
| Fully connected		| 84X43        									|
| Softmax				| 43        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer that minimizes a loss function defined as the mean cross entropy between predicted class probabilities and true labels. The optimization procedure prosceeds as follows. The training data is divided into small 128-example batches. For each batch, the optimizer is invoked to predict the probabilities of the 128 traffic signds and the error between the predictions and true labels are used to compute the gradient. The gradient is then applied by the optimizer to update the parameters of the model. The magintue of the update is a function of the magnitude of the gradient and the learning rate which I set to 0.001. The next batch is then passed to the optimizer to perform the same forward and backward passes of prediction gradient back probagation. After completeing the last batch, the entire process (epoch) is repeated on the whole dataset over and over again. I trained the network for a total of 150 epochs.     

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?
Upond completing the 150-th training epoch, my final model results were:
* training set accuracy of 1.0 (i.e. 100%)
* Validation set accuracy of 0.945.
* test set accuracy of 0.932.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose five random traffic sign images from the test data and I obtained 100% accuracy. 

The images are shown in the solution's notebook.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

28	Children crossing
24	24	Road narrows on the right
13	13	Yield
18	18	General caution
8	8	Speed limit (120km/h)

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing      		| Children crossing					| 
| Road narrows on the right     			| Road narrows on the right	|
| Yeild     			| Yield 										|
| General caution					| General caution				|
| Speed limit (120km/h)	      		| Speed limit (120km/h)				|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is certain that this is a Children Crossing sign (probability of 1.0), and the image does contain a Children Crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00    		| Children crossing					| 
| 0.99     			| Road narrows on the right	|
| 1.00     			| Yield 										|
| 1.00					| General caution				|
| 1.00      		| Speed limit (120km/h)				|



