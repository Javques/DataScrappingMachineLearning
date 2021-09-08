#-------------------------------------------------------
#Assignment 2
#Written by Alexis Bolduc 40126092
#For COMP 472 Section AK-X –Summer 2021
#--------------------------------------------------------
#this program will scrape the webpages of all episode reviews, in order to build a probabilistic model, with the goal to predict if a review from the train data set is negative 
#or positive
#then I change the smoothing factor to see if anything changes

import numpy as np
import string
import pandas as pd
import matplotlib.pyplot as plt
from requests import get
from bs4 import BeautifulSoup

print("Program Running")
nameOfCols = ["Name", "Season", "Review Link", "Year"]
#read list of episode links
episodeList = pd.read_csv("data.csv")
#review infos
titles = []
ratings = []
isItPositive = []
comment = []

#all the review links
reviewLinks = episodeList[nameOfCols[2]].tolist()

#AccessAllWebPages and Parse data into different arrays
def getAllReviews(reviewLink):
    
    for reviewLinkURL in reviewLink:
        
         response = get(reviewLinkURL)
         html_soup = BeautifulSoup(response.text,'html.parser')
         allReviews = html_soup.find_all('div',class_='lister-item-content')
        
         #check if spoiler
         #check if no score
         #get score
         #get review
         
         for review in allReviews:
         #add text to array
         #check for score
            content = review.find('div', class_='content')

            if(review.find('span', class_='point-scale') is not None):
                #ratings
                rating = review.find('span', class_='rating-other-user-rating').span.text
                
                ratings.append(int(rating))
                #isPositive
                isItPositive.append(int(rating)>=8)
                #the Comment
                onlyContent = content.find('div',class_='text show-more__control').text
                comment.append(onlyContent)
                #the title of the review
                theTitle = review.find('a',class_="title").text
                titles.append(theTitle)
         


getAllReviews(reviewLinks)

#creating main data frame
episode_df=pd.DataFrame({'Title':titles,'rating score':ratings,
'IsItPositive': isItPositive,
'Review':comment 
}) 

#creating data.csv
episode_df.to_csv('Reviews.csv')
#creating groups of samples
transState = episode_df.groupby (episode_df.IsItPositive)
AllPos = transState.get_group(True)
AllNeg = transState.get_group(False)

numberOfPosReviews = len(AllPos)
numberOfNegReviews = len(AllNeg)
print("Number of Positive Reviews:")
print(numberOfPosReviews)
print("Number of Negative Reviews:")
print(numberOfNegReviews)
#dividing train set and data set
trainPos = AllPos.sample(frac=0.5,random_state=200)
trainNeg = AllNeg.sample(frac=0.5,random_state=200)
testPos = AllPos.drop(trainPos.index)
testNeg = AllNeg.drop(trainNeg.index)

testDataFrame = testPos.append(testNeg)
print("Number of Train Positive Reviews:")
print(len(trainPos))
print("Number of Train Negative Reviews:")
print(len(trainNeg))
print("Number of Test Positive Reviews:")
print(len(testPos))
print("Number of Test Negative Reviews:")
print(len(testNeg))
#reading stopwords file
theFile = open("remove.txt","r")
#creating file with all the removed words
removed = open("ActuallyRemoved.txt", "w")

#stopwords
stops = theFile.read()
stops = stops.split()



#creating vocabulary
def populateDict(dataTrainSet,dict):
    vocabulary = []
    for Reviews in dataTrainSet['Review']:
        Reviews = str(Reviews)
        Reviews = Reviews.lower()
        table = Reviews.maketrans(dict.fromkeys(string.punctuation))
        Reviews = Reviews.translate(table)
        vocabulary = Reviews.split()
        for word in vocabulary:
        
            if (word in stops ):
                
                removed.write("\n"+word)
            else:
                if(word in dict):
                    dict[word]+=1
                else:
                    dict[word] = 1


vocPositive = dict()
vocNegative = dict()
populateDict(trainNeg,vocNegative)
populateDict(trainPos,vocPositive)    

#get all unique words
allKeys = list(vocPositive.keys())
for key in vocNegative.keys():
    if(key not in allKeys):
        allKeys.append(key)
sizeOfVoc = len(allKeys)
numberInPos = len(vocPositive.keys())
numberInNeg = len(vocNegative.keys())

 
#final voc without duplicates
def populateFinalVoc(smoothFactor):
    theDic = dict()
    for key in allKeys:
        freqPos = 0
        freqNeg = 0
        if(key in vocPositive):
            freqPos = vocPositive[key]
        if(key in vocNegative):
            freqNeg=vocNegative[key]
        Ppos = round(float((freqPos+smoothFactor)/((sizeOfVoc*smoothFactor)+numberInPos)),6)
        Pneg = round(float((freqNeg+smoothFactor)/((sizeOfVoc*smoothFactor)+numberInNeg)),6)
        theDatas = [freqPos,Ppos,freqNeg,Pneg]
        theDic[key] = theDatas
    return theDic

TheUltimateVocabulary = populateFinalVoc(1)



print("Number of unique words:")
print(len(allKeys))

#creating model
Vocab = open("model.txt", "w", encoding="utf-8")
def fileModel(fileReader,theDic):
    count = 0
    for key, value in theDic.items():
        count+=1
        stringToWrite = f"No.{count} {key} \n{value[0]}, {value[1]}, {value[2]}, {value[3]}\n"
        fileReader.write(stringToWrite)
    fileReader.close()

fileModel(Vocab,TheUltimateVocabulary)


#calculating positive review score
def calculatePosScore(arrayOfWord,theDic):
    probOfPos = numberOfPosReviews/(numberOfNegReviews+numberOfPosReviews)
    totalScore = 0
    totalScore+=np.log10(probOfPos)
    for word in arrayOfWord:
        if(word in theDic):
            currentPpos = theDic[word][1]
            totalScore+=np.log10(currentPpos)
    return totalScore
#calculating negative review score
def calculateNegScore(arrayOfWord,theDic):
    probOfNeg = numberOfNegReviews/(numberOfPosReviews+numberOfNegReviews)
    totalScore =0
    totalScore+=np.log10(probOfNeg)
    for word in arrayOfWord:
        if(word in theDic):
            currentPneg = theDic[word][3]
            totalScore+= np.log10(currentPneg)
    return totalScore
#test model with test set
#correctness calculated in f-means in regards to positive reviews
def testDataSet(testSet):
    TestResult = open("result.txt", "w", encoding="utf-8")
    numberOfWrong = 0
    numberOfTest = 0
    A = 0
    B=0
    D=0
    for ind in testSet.index:
        
        currentReview = []
        theReview = testSet['Review'][ind]
        theReview = theReview.lower()
        table = theReview.maketrans(dict.fromkeys(string.punctuation))
        theReview = theReview.translate(table)
        currentReview = theReview.split()
        numberOfTest+=1
        reviewName = testSet['Title'][ind]

        Ppos = calculatePosScore(currentReview,TheUltimateVocabulary)
        Pneg = calculateNegScore(currentReview,TheUltimateVocabulary)

        Result= Ppos>Pneg
        ResultString = "Positive" if Result else "Negative"

        CorrectResult = testSet['IsItPositive'][ind]
        CorrectResultString = "Positive" if CorrectResult else "Negative"

        if(Result):
            if(CorrectResult):
                A+=1
            else:
                B+=1
        else:
            if(CorrectResult):
                D+=1
        PredictionRight = Result==CorrectResult
        PredictionRightString = "Right" if PredictionRight else "Wrong"
        stringResult = f"\nNo.{numberOfTest} {reviewName} {Ppos}, {Pneg}, {ResultString}, {CorrectResultString}, {PredictionRightString}\n"
        TestResult.write(stringResult)
        if (not PredictionRight):
            numberOfWrong+=1

    precision = float(A/(A+B))
    recall = float(A/(A+D))
    fmeans = float(((2*precision*recall)/(precision+recall))*100)
    accuracy = float(((numberOfTest-numberOfWrong)/numberOfTest)*100)
    stringCorrectness = f"The prediction accuracy is {accuracy}% and the F1-measure is {fmeans}" 
    TestResult.write(stringCorrectness)
    TestResult.close()

    
testDataSet(testDataFrame)
#2.2
smoothSteps = [1,1.2,1.4,1.6,1.8,2]
#getting correctness with different smoothing values
def smoothFiltering(steps,testSet):
    smoothFmeans =[]
    A = 0
    B=0
    D=0
    for smoothFactor in steps:
        currentDic = populateFinalVoc(smoothFactor)
        numberOfWrong = 0
        numberOfTest = 0
        if(smoothFactor == 1.6):
            SmoothModel = open("smooth-model.txt", "w", encoding="utf-8")
            fileModel(SmoothModel,currentDic)
            SmoothResult = open("smooth-result.txt", "w", encoding="utf-8")
        for ind in testSet.index:
        
            currentReview = []
            theReview = testSet['Review'][ind]
            theReview = theReview.lower()
            table = theReview.maketrans(dict.fromkeys(string.punctuation))
            theReview = theReview.translate(table)
            currentReview = theReview.split()
            numberOfTest+=1
            reviewName = testSet['Title'][ind]

            Ppos = calculatePosScore(currentReview,currentDic)
            Pneg = calculateNegScore(currentReview,currentDic)

            Result= Ppos>Pneg
            ResultString = "Positive" if Result else "Negative"

            CorrectResult = testSet['IsItPositive'][ind]
            CorrectResultString = "Positive" if CorrectResult else "Negative"

            PredictionRight = Result==CorrectResult
            PredictionRightString = "Right" if PredictionRight else "Wrong"
            if(Result):
                if(CorrectResult):
                    A+=1
                else:
                    B+=1
            else:
                if(CorrectResult):
                    D+=1
            stringResult = f"\nNo.{numberOfTest} {reviewName} {Ppos}, {Pneg}, {ResultString}, {CorrectResultString}, {PredictionRightString}\n"
            if(smoothFactor == 1.6):
                SmoothResult.write(stringResult)
            if (not PredictionRight):
                numberOfWrong+=1
        
        precision = float(A/(A+B))
        recall = float(A/(A+D))
        fmeans = float(((2*precision*recall)/(precision+recall))*100)
        accuracy = float(((numberOfTest-numberOfWrong)/numberOfTest)*100)
        smoothFmeans.append(fmeans)
        stringCorrectness = f"The prediction accuracy is {accuracy}% and the F1-measure is {fmeans}" 
        if(smoothFactor == 1.6):
            SmoothResult.write(stringCorrectness)
            SmoothResult.close()
    return smoothFmeans
Yaxis = smoothFiltering(smoothSteps,testDataFrame)

#Plotting of correctness values over different smoothing values
plt.plot(smoothSteps,Yaxis)
plt.xlabel("smoothing factor")
plt.ylabel("F1-measure of prediction")
plt.title("Correctness of prediction over different smoothing factors")
print("Program Ended")
plt.show()


#end of program thank you!
