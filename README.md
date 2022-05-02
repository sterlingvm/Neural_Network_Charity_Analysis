# Neural Network Charity Analysis

## Analysis Overview
In this project, we want to use a TensorFlow deep-learning neural network to classify and analyze the success and effectiveness of charitable donations to philanthropic organizations.\

The following methods and processes were utilized for the building of our neural network model and the analysis:
- Data Preprocessing via OneHot-Encoding, StandardScaler, Train_Test_Split, and Pandas DataFrame Manipulation in Python
- Model building, compilation, training, evaluation, and optimization via TensorFlow.Keras
- Model checkpointing, exporting, callbacks, and classifier weight saving via TensorFlow's `ModelCheckpoint`

## Resources
- Data Source: [charity_data.csv](https://github.com/sterlingvm/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)
- Software: Python 3.8.8, Visual Studio Code 1.66.2, Jupyter Notebook 6.4.5, Conda 4.12.0

## Results

### Data Preprocessing
- The categorical columns `EIN` and `NAME` are columns with impertinent information for our model information so we've removed them from the input data.
We have encoded all pertinent, non-catigorical data into numerical data so that our model can interpret and utilize this data.

- The `IS_SUCCESSFUL` column contains binary data with outcomes of charity donation effectiveness (our target metric). Thus, this variable serves as the target for our deep learning neural network.
Using `IS_SUCCESSFUL` as our y target group, we proceeded to split our data into training and testing datasets for the model

- The following columns `APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT` are the features that we will utilize in our model.\
With these features in our X dataset, we proceeded to standardize values as to ensure that our model would not skew the weight or importance of certain feature values innaccurately and inappropriately.

### Compiling, Training, and Evaluating the Model
- This deep-learning neural network model is comprised of two hidden layers with 90 and 40 neurons respectively.\
The input data has 114 features and 34,299 samples.\
To optimize speed and accuracy for the model's training process, we implemented  `ReLU` activation functions for the 2 hidden layers. Our output is a binary classification, so we used `Sigmoid` in our output layer.\
For the compilation, the optimizer is `adam` and the loss function is `binary_crossentropy`.

- The model accuracy is under 75%. This is not a satisfying performance to help predict the outcome of the charity donations.
- To increase the performance of the model, we applied bucketing to the feature `ASK_AMT` to organize the different values by intervals. We also implemented a third hidden layer, increased the neurons in each layer to 100, 35, and 10, and change the number of epochs for the training from 100 to 150.\

In many attempts at optimization we used different amounts of neurons, different durations of epochs, and also tried a different activation function (`tanh`) but none of these steps helped improve the model's overall performance to surpass 75%.

## Summary
We did not reach the target accuracy of 75%+ with out deep learning neural network model. Our model isn't unusable for predicting the effectiveness of charity donations, but it isn't as accurate as we'd have liked it to be.\
As our result was a binary classification situation, another option for statistical modelling would be a supervised machine learning model such as the Random Forest Classifier. With a Random Forest Classifier (for example) we could combine a multitude of decision trees to generate a classified output and evaluate its performance against the deep learning neural network model.