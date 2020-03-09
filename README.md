# Face-And-Digit-Classification
# Implement two classiﬁcation algorithms for detecting faces and classifying digits: 
Perceptron & Naive Bayes Classiﬁer 
# Design the features for each of the two problems, and write a program for extracting the features from each image.
Looked at the pixels surrounding the current pixel and increased the probability if the pixels formed a certain pattern.
For example, if the shape formed by the pixels was diagonal, it is more likely to be a curve, like an 2,3,5,6,8,9,0.
# Train the two algorithms on the part of the data set that is reserved for training. 
First,useonly 10% ofthedatapoints that are reserved for training, then 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, and ﬁnally 100%. 
The results are the output of a function of with input the percentage of data points used for training. 
# Compare the performances of the two algorithms using the part of the dataset that is reserved for testing, and report:
• The time needed for training as a function of the number (or percentage) of data points used for training. 
• The prediction error (and standard deviation) as a function of the number (or percentage) of data points used for training. 
• This can be seen in the report
