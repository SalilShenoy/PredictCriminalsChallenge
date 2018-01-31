Predict Criminal Challenge

I read the training data, in a pandas data frame and then used correlation matrix to figure out the features
which affected the target (Criminal) variable.

Before, calculating the correlation I filtered out the missing values. From the discussion forum, I got to know that
missing values in the data are represented by -1.

Having filtered the data and figured out the features which I need to look at closely from the correlation, I created
dataframes from the feature specific data. This is just to look at and visualize the data.

I went ahead by using a Descision Tree Classifier for the data using the default parameters and split of 100. I improved
model by checking the precision and r2_score which I have commented out in the final submission.

To build the model I split the training data using train_test_split (85:15) and eliminated features which were hampering
precision and r2_score.

