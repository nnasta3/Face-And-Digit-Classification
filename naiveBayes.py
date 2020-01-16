# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters. The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    "*** YOUR CODE HERE ***"

    #print(trainingData[0])
    #print(trainingLabels)
    #print(kgrid)
   
    
    #Create a data struct (mainCounter) to keep track of all features for every label and increment the count
    #Struct looks like this: ((feature x, feature y), label): [count if value @feature is 0, count if value @feature is 1]
    #values for if feature is 0 and 1 are stored as a list for each feature,label pair, the key is the feature,label pair and the value is the list of counts 
    mainCounter = util.Counter()
    for label in self.legalLabels:
      for feature in self.features:
        mainCounter[(feature, label)] = [0,0]

    #count every feature,label pair and increment the coresponding index in list of values
    for i in range(len(trainingData)):
      currData = trainingData[i]
      label = trainingLabels[i]
      for feature in currData.keys():
        #This is only for enhanced feature weight calculation
        if currData[feature] == 5:
          mainCounter[(feature, label)][1] += 5
        else:
          mainCounter[(feature, label)][currData[feature]] += 1

    #Calculate prior distribution P(Y)
    #Number of training instances with label y / total number of training instances n => y/n == prior distribution 
    #Need to count each time a label is seen, can use util.normalize after counting all instances with label y
    priorDist = util.Counter()
    for i in range(len(trainingData)):
      label = trainingLabels[i]
      priorDist[label] += 1
        #print(featureValues[feature])
    self.priorDist = util.normalize(priorDist)
    #print(mainCounter)

    #use mainCount to compute probabilities for each feature, label pair
    mainCounterProbs = util.Counter()
    for label in self.legalLabels:
      for feature in self.features:
        mainCounterProbs[feature, label] = util.normalize(mainCounter[feature, label])

    #print(conditionalProbabilities)
    self.mainCounterProbs = mainCounterProbs
    
    #util.raiseNotDefined()
    '''
    #THIS CODE DOES NOT WORK GREAT, ONLY PRODUCES ~ 20% WITH FULL TEST DATA 
    #FAILED ATTEMPT :(

    labelsCount = util.Counter()
    for label in trainingLabels:
      labelsCount[label] += 1

    #print(labelsCount)
    priorDist = util.Counter()
    #priorDist = util.normalize(labelsCount)
    for label in labelsCount:
      priorDist[label] = labelsCount[label] / len(trainingLabels)
    #print("Prior Dist",priorDist)

    #Calculate conditional probabilities for each feature,label pair
    #For each label y, keep track of all features and increment a feature if it is positive (>0) 
    
    trueFeatureCount = util.Counter()
    totalFeatureCount = util.Counter()
    falseFeatureCount = util.Counter()
    #print (trueFeatureCount)
    for i in range(len(trainingData)):
      currData = trainingData[i]
      currLabel = trainingLabels[i]
      #print(trainingData[currData])
      #value > 0 = pixel is a '+' or '#'
      for feature, value in currData.items():
        #print(trainingData[currData].items())
        #print(feature)
        #print(value)
        totalFeatureCount[( feature, currLabel )] += 1
        if value > 0:
          #print(feature,value)
          trueFeatureCount[( feature, currLabel )] += 1
          #print(trueFeatureCount[( (feature), trainingLabels[currData] )])
        else:
          falseFeatureCount[((feature), currLabel)] += 1
          
    #print(totalFeatureCount) 
    featureProbTrue = util.Counter()
    featureProbFalse = util.Counter()
    
    #smoothing
    for label in self.legalLabels:
      for feature in self.features:
        totalFeatureCount[(feature,label)] += 4.0
        if (feature,label) in featureProbTrue.keys():
          featureProbTrue[(feature,label)] += 2.0
        elif (feature,label) in featureProbFalse.keys():
          featureProbFalse[(feature,label)] += 2.0

    
    for feature, value in trueFeatureCount.items():
      featureProbTrue[feature] = (trueFeatureCount[feature]  ) / (totalFeatureCount[feature])
     
    for feature, value in falseFeatureCount.items():
      featureProbFalse[feature] = (falseFeatureCount[feature]  ) / (totalFeatureCount[feature] )
    
    #print("TRUE",featureProbTrue)

    #print("FALSE",featureProbFalse)

    self.labelsProb = priorDist
    self.featureProbTrue = featureProbTrue
    self.featureProbFalse = featureProbFalse
    #util.raiseNotDefined()
    '''
    
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"

    '''
    #PART OF FAILED CODE ABOVE

    for label in self.legalLabels:
      #print(self.labelsProb[label])
      logJoint[label] =  self.labelsProb[label]
      for feature, value in datum.items():
        if value > 0:
          logJoint[label] *=  self.featureProbTrue[feature,value]
        else:
          logJoint[label] *=  self.featureProbFalse[feature,value]
    '''
    #For every feature,label pair being tested, check if the value is >0 or ==0 and use correspoding index in the mainCounter list
    for label in self.legalLabels:
      logJoint[label] = self.priorDist[label]
      for key, value in datum.items():
        if value > 0:
          logJoint[label] *= self.mainCounterProbs[key, label][1]
        else:
          logJoint[label] *= self.mainCounterProbs[key, label][0]

    #print(logJoint)
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
