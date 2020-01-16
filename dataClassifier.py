# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

import time
import mostFrequent
import naiveBayes
import perceptron
import mira
import samples
import sys
import util
import math
import random

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.
  
  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## DESCRIBE YOUR ENHANCED FEATURES HERE...

 
  ##
  """
  features =  basicFeatureExtractorDigit(datum)
  
  "*** YOUR CODE HERE ***"  
  ###########################################################################################################################################################################################################

  #print(features)
  #My feature looks at pixels, if the current pixel is black and is surrounded by other pixels that are black, then increase the weight of the current pixel by 5
  #otherwise keep the current weight of 1. 
  newFeatures = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if features[(x, y)] > 0:
        if x-1 > 0 and y-1 > 0 and x+1 <28 and y+1 <28:
          if features[(x-1, y-1)] == 1 and features[(x, y-1)] == 1 and features[(x-1, y+1)] == 1 and features[(x-1, y)] == 1 and features[(x+1, y)] == 1 and features[(x+1, y-1)] == 1 and features[(x+1, y)] == 1 and features[(x+1, y+1)] == 1:
            newFeatures[(x,y)] = 5
          else:
            newFeatures[(x,y)] = 1
      else:
        newFeatures[(x,y)] = 0

  #print(newFeatures)

  #####################################################################################################################################################################################################################
  return newFeatures


def contestFeatureExtractorDigit(datum):
  """
  Specify features to use for the minicontest
  """
  features =  basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  features =  basicFeatureExtractorFace(datum)
  ###########################################################################################################################################################################################################

  #print(features)
  #My feature looks at pixels, if the current pixel is black and is surrounded by other pixels that are black, then increase the weight of the current pixel by 5
  #otherwise keep the current weight of 1. 
  newFeatures = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if features[(x, y)] > 0:
        if x-1 > 0 and y-1 > 0 and x+1 <28 and y+1 <28:
          if features[(x-1, y-1)] == 1 or features[(x, y-1)] == 1 or features[(x-1, y+1)] == 1 or features[(x-1, y)] == 1 or features[(x+1, y)] == 1 or features[(x+1, y-1)] == 1 or features[(x+1, y)] == 1 or features[(x+1, y+1)] == 1:
            newFeatures[(x,y)] = 5
          else:
            newFeatures[(x,y)] = 1
      else:
        newFeatures[(x,y)] = 0

  #print(newFeatures)

  #####################################################################################################################################################################################################################

  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print ("===================================")
          print ("Mistake on example %d" % i) 
          print ("Predicted %d; truth is %d" % (prediction, truth))
          print ("Image: ")
          print (rawTestData[i])
          break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print ("new features:", pix)
            continue
      print (image)  

def default(str):
  return str + ' [Default: %default]'

# Main harness code
def runClassifier( ):
  ########################################################################################################################################
  #Edited Code
  #Store info for each iteration
  nbDigits = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  nbFaces = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  perceptronDigits = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  perceptronFaces = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  trainingCounts = {0: 500, 1: 1000, 2: 1500, 3: 2000, 4: 2500, 5: 3000, 6: 3500, 7: 4000, 8: 4500, 9:5000, 10: 500, 11: 1000, 12: 1500, 13: 2000, 14: 2500, 15: 3000, 16: 3500, 17: 4000, 18: 4500, 19:5000,
                    20: 45, 21: 90, 22: 135, 23: 180, 24: 225, 25: 270, 26: 315, 27: 360, 28: 405, 29: 450,  30: 45, 31: 90, 32: 135, 33: 180, 34: 225, 35: 270, 36: 315, 37: 360, 38: 405, 39: 450}
  #FaceData
  rawFaceTrainingData = samples.loadDataFile("facedata/facedatatrain", 450,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
  faceTrainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", 450)

  rawFaceValidationData = samples.loadDataFile("facedata/facedatatrain", 300,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
  faceValidationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", 300)

  rawFaceTestData = samples.loadDataFile("facedata/facedatatest", 149,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
  testFaceLabels = samples.loadLabelsFile("facedata/facedatatestlabels", 149)

  #DigitData
  rawDigitTrainingData = samples.loadDataFile("digitdata/trainingimages", 5000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
  digitTrainingLabels = samples.loadLabelsFile("digitdata/traininglabels", 5000)

  rawDigitValidationData = samples.loadDataFile("digitdata/validationimages", 1000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
  digitValidationLabels = samples.loadLabelsFile("digitdata/validationlabels", 1000)

  rawDigitTestData = samples.loadDataFile("digitdata/testimages", 1000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
  testDigitLabels = samples.loadLabelsFile("digitdata/testlabels", 1000)
      
  #Automation of test for each classifier and data type
  for x in range(40):
    

    if x<10:
      classifierName = "nb"
    elif x<20:
      classifierName = "perceptron"
    elif x<30:
      classifierName = "nb"
    else:
      classifierName = "perceptron"

    if x < 20:
      Data = "digits"
    else:
      Data = "faces"

    if(Data=="digits"):
      legalLabels = range(10)
      #featureFunction = enhancedFeatureExtractorDigit
      featureFunction = basicFeatureExtractorDigit
    else:
      legalLabels = range(2)
      #featureFunction = enhancedFeatureExtractorFace
      featureFunction = basicFeatureExtractorFace

    if(classifierName == "nb"):
      classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
      classifier.setSmoothing(2.0)

    elif(classifierName == "perceptron"):
      classifier = perceptron.PerceptronClassifier(legalLabels,3)

    print ("Doing classification")
    print ("--------------------")
    print ("data:\t\t" + Data)
    print ("classifier:\t\t " + classifierName)
    print ("using enhanced features")
    print ("training set size:\t" + str(trainingCounts[x]))
    
    
    # Extract features
    print ("Extracting features...")
    # Load data  
    
    
    if Data == "digits":
      startTime = time.process_time()
      h = 0
      while h <3:
        print("Iteration %d" % h)
        numTraining = trainingCounts[x]
        rawTrainingData = []
        rawTrainingLabels = []
        i=0
        while i < numTraining:
          k = list(range(0,5000))
          random.shuffle(k)
          j = k.pop()
          rawTrainingLabels.append(digitTrainingLabels[j])
          rawTrainingData.append(rawDigitTrainingData[j])
          i+=1

        trainingData = list(map(featureFunction, rawTrainingData))
        validationData = list(map(featureFunction, rawDigitValidationData))
        testData = list(map(featureFunction, rawDigitTestData))

        print ("Training...")
        classifier.train(trainingData, rawTrainingLabels, validationData, digitValidationLabels)
        print ("Validating...")
        guesses = classifier.classify(validationData)
        correct = [guesses[i] == digitValidationLabels[i] for i in range(len(digitValidationLabels))].count(True)
        print (str(correct), ("correct out of " + str(len(digitValidationLabels)) + " (%.1f%%).") % (100.0 * correct / len(digitValidationLabels)))
        print ("Testing...")
        guesses = classifier.classify(testData)
        correct = [guesses[i] == testDigitLabels[i] for i in range(len(testDigitLabels))].count(True)
        print (str(correct), ("correct out of " + str(len(testDigitLabels)) + " (%.1f%%).") % (100.0 * correct / len(testDigitLabels)))
        h+=1
        #Gather correct count for each iteration and use to compute standard deviation
        if classifierName == "nb":
          if Data == "digits":
            nbDigits[x%10] += correct
            nbDigits[(x%10) + 10] += time.process_time() - startTime
          else:
            nbFaces[x%10] += correct
            nbFaces[(x%10) + 10] += time.process_time() - startTime
        else:
          if Data == "digits":
            perceptronDigits[x%10] += correct
            perceptronDigits[(x%10) + 10] += time.process_time() - startTime
          else:
            perceptronFaces[x%10] += correct
            perceptronFaces[(x%10) + 10] += time.process_time() - startTime
    else:
      h =0
      while h<3:
        print("Iteration %d" % h)
        numTraining = trainingCounts[x]
        rawTrainingData = []
        rawTrainingLabels = []
        i=0
        while i < numTraining:
          k = list(range(0,450))
          random.shuffle(k)
          j=k.pop()
          rawTrainingLabels.append(faceTrainingLabels[j])
          rawTrainingData.append(rawFaceTrainingData[j])
          i+=1
        trainingData = list(map(featureFunction, rawTrainingData))
        validationData = list(map(featureFunction, rawFaceValidationData))
        testData = list(map(featureFunction, rawFaceTestData))

        print ("Training...")
        classifier.train(trainingData, rawTrainingLabels, validationData, faceValidationLabels)
        print ("Validating...")
        guesses = classifier.classify(validationData)
        correct = [guesses[i] == faceValidationLabels[i] for i in range(len(faceValidationLabels))].count(True)
        print (str(correct), ("correct out of " + str(len(faceValidationLabels)) + " (%.1f%%).") % (100.0 * correct / len(faceValidationLabels)))
        print ("Testing...")
        guesses = classifier.classify(testData)
        correct = [guesses[i] == testFaceLabels[i] for i in range(len(testFaceLabels))].count(True)
        print (str(correct), ("correct out of " + str(len(testFaceLabels)) + " (%.1f%%).") % (100.0 * correct / len(testFaceLabels)))
        h+=1
        #Gather correct count for each iteration and use to compute standard deviation
        if classifierName == "nb":
          if Data == "digits":
            nbDigits[x%10] += correct
            nbDigits[(x%10) + 10] += time.process_time() - startTime
          else:
            nbFaces[x%10] += correct
            nbFaces[(x%10) + 10] += time.process_time() - startTime
        else:
          if Data == "digits":
            perceptronDigits[x%10] += correct
            perceptronDigits[(x%10) + 10] += time.process_time() - startTime
          else:
            perceptronFaces[x%10] += correct
            perceptronFaces[(x%10) + 10] += time.process_time() - startTime

    
  #NAIVE BAYES DIGITS
  print("Average Correct Guesses for Naive Bayes Digits Based on Percentage of TrainingData Used")
  print("10%% %d/1000, 20%% %d/1000, 30%% %d/1000, 40%% %d/1000, 50%% %d/1000, 60%% %d/1000, 70%% %d/1000, 80%% %d/1000, 90%% %d/1000, 100%% %d/1000" % (nbDigits[0]/3,nbDigits[1]/3,nbDigits[2]/3,nbDigits[3]/3,nbDigits[4]/3,nbDigits[5]/3,nbDigits[6]/3,nbDigits[7]/3,nbDigits[8]/3,nbDigits[9]/3))
  print("Standard Deviation for Naive Bayes Digits Based on Percentage of Training Data Used")
  stndDev = [0,0,0,0,0,0,0,0,0,0]
  i=0
  while i <10:
    stndDev[i] = nbDigits[i]/3
    stndDev[i] = nbDigits[i] - stndDev[i]
    stndDev[i] = math.pow(stndDev[i],2)
    stndDev[i] = stndDev[i] / 1000
    stndDev[i] = math.sqrt(stndDev[i])
    i+=1
  print("10%% %d, 20%% %d, 30%% %d, 40%% %d, 50%% %d, 60%% %d, 70%% %d, 80%% %d, 90%% %d, 100%% %d" % (stndDev[0],stndDev[1],stndDev[2],stndDev[3],stndDev[4],stndDev[5],stndDev[6],stndDev[7],stndDev[8],stndDev[9]))
  print("Average Time to Complete Each Iteration Based on Percentage of Training Data Used In Seconds")
  print("10%% %d seconds, 20%% %d seconds, 30%% %d seconds, 40%% %d seconds, 50%% %d seconds, 60%% %d seconds, 70%% %d seconds, 80%% %d seconds, 90%% %d seconds, 100%% %d seconds" % (nbDigits[10] / 3,nbDigits[11]/3,nbDigits[12]/3,nbDigits[13]/3,nbDigits[14]/3,nbDigits[15]/3,nbDigits[16]/3,nbDigits[17]/3,nbDigits[18]/3,nbDigits[19]/3))
 
  #NAIVE BAYES FACES
  print("Average Correct Guesses for Naive Bayes Faces Based on Percentage of TrainingData Used")
  print("10%% %d/149, 20%% %d/149, 30%% %d/149, 40%% %d/149, 50%% %d/149, 60%% %d/149, 70%% %d/149, 80%% %d/149, 90%% %d/149, 100%% %d/149" % (nbFaces[0]/3,nbFaces[1]/3,nbFaces[2]/3,nbFaces[3]/3,nbFaces[4]/3,nbFaces[5]/3,nbFaces[6]/3,nbFaces[7]/3,nbFaces[8]/3,nbFaces[9]/3))
  stndDev = [0,0,0,0,0,0,0,0,0,0]
  i=0
  while i <10:
    stndDev[i] = nbFaces[i]/3
    stndDev[i] = nbFaces[i] - stndDev[i]
    stndDev[i] = math.pow(stndDev[i],2)
    stndDev[i] = stndDev[i] / 149
    stndDev[i] = math.sqrt(stndDev[i])
    i+=1
  print("Standard Deviation for Naive Bayes Faces Based on Percentage of Training Data Used")
  print("10%% %d, 20%% %d, 30%% %d, 40%% %d, 50%% %d, 60%% %d, 70%% %d, 80%% %d, 90%% %d, 100%% %d" % (stndDev[0],stndDev[1],stndDev[2],stndDev[3],stndDev[4],stndDev[5],stndDev[6],stndDev[7],stndDev[8],stndDev[9]))
  print("Time to Complete Each Iteration Based on Percentage of Training Data Used In Seconds")
  print("10%% %d, 20%% %d, 30%% %d, 40%% %d, 50%% %d, 60%% %d, 70%% %d, 80%% %d, 90%% %d, 100%% %d" % (nbFaces[10]/3,nbFaces[11]/3,nbFaces[12]/3,nbFaces[13]/3,nbFaces[14]/3,nbFaces[15]/3,nbFaces[16]/3,nbFaces[17]/3,nbFaces[18]/3,nbFaces[19]/3))

  #PERCEPTRON DIGITS
  print("Average Correct Guesses for Perceptron Digits Based on Percentage of Training Data Used")
  print("10%% %d/1000, 20%% %d/1000, 30%% %d/1000, 40%% %d/1000, 50%% %d/1000, 60%% %d/1000, 70%% %d/1000, 80%% %d/1000, 90%% %d/1000, 100%% %d/1000" % (perceptronDigits[0]/3,perceptronDigits[1]/3,perceptronDigits[2]/3,perceptronDigits[3]/3,perceptronDigits[4]/3,perceptronDigits[5]/3,perceptronDigits[6]/3,perceptronDigits[7]/3,perceptronDigits[8]/3,perceptronDigits[9]/3))
  print("Standard Deviation for Perceptron Digits Based on Percentage of Training Data Used")
  stndDev = [0,0,0,0,0,0,0,0,0,0]
  i=0
  while i <10:
    stndDev[i] = perceptronDigits[i]/3
    stndDev[i] = perceptronDigits[i] - stndDev[i]
    stndDev[i] = math.pow(stndDev[i],2)
    stndDev[i] = stndDev[i] / 1000
    stndDev[i] = math.sqrt(stndDev[i])
    i+=1
  print("10%% %d, 20%% %d, 30%% %d, 40%% %d, 50%% %d, 60%% %d, 70%% %d, 80%% %d, 90%% %d, 100%% %d" % (stndDev[0],stndDev[1],stndDev[2],stndDev[3],stndDev[4],stndDev[5],stndDev[6],stndDev[7],stndDev[8],stndDev[9]))
  print("Time to Complete Each Iteration Based on Percentage of Training Data Used In Seconds")
  print("10%% %d, 20%% %d, 30%% %d, 40%% %d, 50%% %d, 60%% %d, 70%% %d, 80%% %d, 90%% %d, 100%% %d" % (perceptronDigits[10]/3,perceptronDigits[11]/3,perceptronDigits[12]/3,perceptronDigits[13]/3,perceptronDigits[14]/3,perceptronDigits[15]/3,perceptronDigits[16]/3,perceptronDigits[17]/3,perceptronDigits[18]/3,perceptronDigits[19]/3))

  #PERCEPTRON FACES
  print("Average Correct Guesses for Perceptron Faces Based on Percentage of Training Data Used")
  print("10%% %d/149, 20%% %d/149, 30%% %d/149, 40%% %d/149, 50%% %d/149, 60%% %d/149, 70%% %d/149, 80%% %d/149, 90%% %d/149, 100%% %d/149" % (perceptronFaces[0]/3,perceptronFaces[1]/3,perceptronFaces[2]/3,perceptronFaces[3]/3,perceptronFaces[4]/3,perceptronFaces[5]/3,perceptronFaces[6]/3,perceptronFaces[7]/3,perceptronFaces[8]/3,perceptronFaces[9]/3))
  print("Standard Deviation for Perceptron Faces Based on Percentage of Training Data Used")
  stndDev = [0,0,0,0,0,0,0,0,0,0]
  i=0
  while i <10:
    stndDev[i] = perceptronFaces[i]/3
    stndDev[i] = perceptronFaces[i] - stndDev[i]
    stndDev[i] = math.pow(stndDev[i],2)
    stndDev[i] = stndDev[i] / 149
    stndDev[i] = math.sqrt(stndDev[i])
    i+=1
  print("10%% %d, 20%% %d, 30%% %d, 40%% %d, 50%% %d, 60%% %d, 70%% %d, 80%% %d, 90%% %d, 100%% %d" % (stndDev[0],stndDev[1],stndDev[2],stndDev[3],stndDev[4],stndDev[5],stndDev[6],stndDev[7],stndDev[8],stndDev[9]))
  print("Time to Complete Each Iteration Based on Percentage of Training Data Used In Seconds")
  print("10%% %d, 20%% %d, 30%% %d, 40%% %d, 50%% %d, 60%% %d, 70%% %d, 80%% %d, 90%% %d, 100%% %d" % (perceptronFaces[10]/3,perceptronFaces[11]/3,perceptronFaces[12]/3,perceptronFaces[13]/3,perceptronFaces[14]/3,perceptronFaces[15]/3,perceptronFaces[16]/3,perceptronFaces[17]/3,perceptronFaces[18]/3,perceptronFaces[19]/3))
    

if __name__ == '__main__':
  # Read input
  #args, options = readCommand( sys.argv[1:] ) 
  # Run classifier
  runClassifier()