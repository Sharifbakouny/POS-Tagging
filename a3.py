import nltk 
import sys
import os
from nltk.tag import brill
from nltk.tag.brill import Template 
from nltk.metrics import accuracy
java_path = "C:/Program Files/Java/jdk-9.0.1/bin"
os.environ['JAVAHOME'] = java_path


from nltk.tag.stanford import StanfordPOSTagger 



def preProcessing(trainingtext):
    
    tupleList = []
    processedList = []
        
    trainingList = trainingtext.split("\n")

    
    #split into newlines
    for eachLine in trainingList:    
        
        if len(eachLine)>1:
            
            splittext = eachLine.split()
            token = splittext[0]
            tag = splittext [1]            
            
            tokentag = (token,tag)
            
            tupleList.append(tokentag)
        
        else:
            processedList.append(tupleList) #append the sentence list to the processed list
            tupleList = [] #reset to empty to allow for next sentence

    
    return processedList

def untagged(processedList):
    untagged = []
    for x in processedList:
        sen = []
        for y in x:
            
            sen.append(y[0])
        untagged.append(sen)
    #print(untagged)
    return untagged
    
def HMMTraining (processedtext):
    
    #adapted from here https://gist.github.com/blumonkey/007955ec2f67119e0909
    
    trainer = nltk.HiddenMarkovModelTrainer() #intialize trainer object
    tagger = trainer.train(labeled_sequences=processedtext) #create HMM model tagger
    
    return tagger


def HMMTesting (testingtext,HMMtagger,processedtext):
    
    HMMtagger.test(testingtext,verbose = True)
    #pass
    

def BrillTraining(processedtext,importedtagger):
        
    #adapted from https://www.geeksforgeeks.org/nlp-brill-tagger/
    templates = [brill.Template(brill.Pos([-1])), brill.Template(brill.Pos([-1]), brill.Word([0]))] 
    
    trainer = nltk.BrillTaggerTrainer(importedtagger,templates,trace=3) 
          
    tagger = trainer.train(processedtext,max_rules=10)
    return tagger
    
def Testing(untaggedList,tagger,processedtext,ifStanford=False):
    res = []
    sentenceTagList = []
    
    if ifStanford == True:
        for sen in untaggedList:
            
            taggedLine = tagger.tag(sen)
            
            for everyTuple in taggedLine:
    
                splittext = everyTuple[1].split('/')
                token = splittext[0]
                tag = splittext [1]            
                
                tokentag = (token,tag)
                
                sentenceTagList.append(tokentag)            
                        
            res.append(sentenceTagList)
            sentenceTagList = []
    
    else:
        for sen in untaggedList:        
            res.append(tagger.tag(sen))

    totalAccuracy = 0    
     
    for i in range(len(res)):
        acc = accuracy(res[i],processedtext[i])
        print("Sentence Number: ", i+1,"/",len(res),"\n")
        print("Test: ","".join(str(processedtext[i])))
        print("\n")
        print("Untagged: ", " ".join(untaggedList[i]))
        print("\n")
        print("Tagged: ", "".join(str(res[i])))
        print("\n")
        print("Accuracy: " , acc)
        print("\n-------------------------------------------\n")
        
        
        #calculation for stanford model
        totalAccuracy = totalAccuracy + acc
        
     
    #Printing Errors  
    print("\n\n\n************************************\n************************************")    
    print("Errors: ")
    print("************************************\n************************************\n\n\n")
    error_counter = 1
    errorlist = []
    errordict = {}
    for i in range(len(res)):
        for j in range(len(res[i])):
            if res[i][j] != processedtext[i][j]:
                print("Error :", error_counter)
                print("Tagged as: " ,res[i][j])
                print("Correct Tag: ", processedtext[i][j])
                errorlist.append((processedtext[i][j][1],res[i][j][1]))
                
                print("\n")
                error_counter+=1
    
    errorlist = sorted(errorlist, key = errorlist.count,reverse = True)
    #print(errorlist)
    
    for error in errorlist:
        errordict[error] = errorlist.count(error)
    print("\n\n\n************************************\n************************************") 
    print("Error Frequency of the tagger, Sorted By Frequency , each tuple is of format (Correct Tag, Wrong Tag): \n")    
    print(errordict)
    print("************************************\n************************************\n\n\n")
    # Finding the most common errors made by the tagger
    
      
    print("\n\n\n************************************\n************************************")
    
    if ifStanford == True:
        print("Tagger Accuracy: " , totalAccuracy/len(res))
    else:
        print("Tagger Accuracy: " ,tagger.evaluate(processedtext))
    print("************************************\n************************************\n\n\n")

def main():
    
    outputtextname = input("Enter output filename (include .txt exension): ")
    modeltype = input("Please enter model type (stanford,hmm,brill) as listed in parentheses: ")    
    
    trainingtextname = input("Enter the TRAINING text filename (include .txt exension): ")
    testingtextname = input("Enter the TESTING text filename (include .txt exension): ")
    filewrite = open(outputtextname, 'w')
    
    orig_stdout = sys.stdout  #print everything that would be printed on the screen to the output file      
    sys.stdout = filewrite
    
    #open the training text
    trainingtext= open(trainingtextname,"r") # the name of the training file
    trainingtext = trainingtext.read()
    
    #open the testing text
    testingtext = open(testingtextname,"r")    
    testingtext = testingtext.read()
    
    processedtext = preProcessing(trainingtext)  #format the training text for the tagger
    
    processedTestText = preProcessing(testingtext) #process test text to be able to test against inputted model
    untaggedList = untagged(processedTestText)
    
    #use stanford tagger
    if modeltype == "stanford":
        _path_to_model = input("Enter the path to your generated tagger (Example: stanford-postagger/Domain2.tagger): ")
        _path_to_jar = input("Enter the path to your .jar file (Example: stanford-postagger/stanford-postagger.jar): ")            

        stanford = StanfordPOSTagger(model_filename=_path_to_model, path_to_jar=_path_to_jar)
        
        print("\n\n\n************************************\n************************************")
        print("Testing Stanford Tagger ... ")
        print("************************************\n************************************\n\n\n")
        
        Testing(untaggedList,stanford,processedTestText,True)
    
    #use hmm tagger
    elif modeltype == "hmm":
        
        HMMtagger = HMMTraining(processedtext) #create model tagger
        
        print("\n\n\n************************************\n************************************")
        print("Testing HMM Tagger ... ")
        print("************************************\n************************************\n\n\n")
        
        Testing(untaggedList,HMMtagger,processedTestText)

   
    #use brill tagger
    elif modeltype == "brill":
        
        #create brill tagger, which requires the hmm tagger to be created
        HMMtagger = HMMTraining(processedtext) #create model tagger
        brillHMMTagger = BrillTraining(processedTestText,HMMtagger)
        
        print("\n\n\n************************************\n************************************")
        print("Testing Brill Tagger ...")
        print("************************************\n************************************\n\n\n")
        
        Testing(untaggedList,brillHMMTagger,processedTestText)


    sys.stdout = orig_stdout
    filewrite.close()
    
main()