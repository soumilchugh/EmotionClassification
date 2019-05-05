from __future__ import division
import nltk
import csv
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import numpy as np
import math
import tensorflow as tf
import time
from nltk.data import find
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

validLabels = list()
validData = list()
reg = 0.1

def convertToOneHot(trainLabels):
    oneHottrainLabels = np.zeros((trainLabels.shape[0],4))
    for item,data in enumerate(trainLabels):
        oneHottrainLabels[item][int(data)] = 1 
    return oneHottrainLabels

def buildGraph(maximum, learning_rate=None, lossType=None, beta1=None, beta2=None, epsilon=None): # Your implementation here
    new_W = tf.Variable(tf.truncated_normal(tf.zeros([maximum,4]).shape, stddev=0.05, dtype=tf.float32, name = "initial_W"))
    b = tf.Variable(0.0,name = "initial_b")
    traindata = tf.placeholder(tf.float32,[None,maximum], name = "trainingdata")
    y_ = tf.matmul(traindata, new_W)
    predicted_outputs = tf.math.add( y_, b, name="predicted_outputs")
    trainlabels = tf.placeholder(tf.float32,[None,4],name = "labels")
    totalLoss = tf.Variable(0.0,name = "loss")
    lamb = tf.placeholder(tf.float32, name = 'lamba')
    tf.set_random_seed(421)
    if lossType == "MSE":
        cost1 = tf.losses.mean_squared_error(labels = trainlabels , predictions = predicted_outputs)
    elif lossType == "CE":
        cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_outputs, labels=trainlabels))
        #cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = predicted_outputs, labels = trainlabels))
    reg = tf.multiply(lamb / 2, tf.reduce_sum(tf.square(new_W)))
    totalLoss = cost1 + reg 
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name='Adam').minimize(totalLoss)  # Initialise the adam optimizer
    return new_W, b, traindata, predicted_outputs, trainlabels, lamb, optimizer, totalLoss

def main():
    trainingLossList = list();
    trainingAccuracyList = list()
    testLossList = list()
    validationLossList = list()
    iterationsList = list()
    validationAccuracyList = list();
    testAccuracyList = list()    
    l = 0
    trainData = list()
    trainLabels = list()
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english')) 
    data = list()
    finaldata = list()
    sentence = list()

    with open("emotions.csv") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        for i, row in enumerate(reader):
            data = list()
            text = row[2]
            sentence = []        
            lower_case = text.lower()
            tokenizer = nltk.tokenize.WordPunctTokenizer()
            tokens = tokenizer.tokenize(lower_case)
            stemmer = nltk.stem.WordNetLemmatizer()
            for token in tokens:
                if token not in stop_words: 
                    sentence.append(stemmer.lemmatize(token))
                #print sentence
            sentence1 = (" ".join(sentence)).strip()    
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            token = tokenizer.tokenize(sentence1)
            if len(tokenizer.tokenize(sentence1)) <= 50:
                trainData.append(sentence1)
                trainLabels.append(row[4])
                for word in token:
                    data.append(word)
                finaldata.append(data)
                if l < len(tokenizer.tokenize(sentence1)):
                    l = len(tokenizer.tokenize(sentence1))
                    maximum = l
                    print (sentence1)
    print (maximum)           
    f.close()
    print (len(trainLabels))
    finaltrainLabels = np.array(trainLabels[:90000])
    finaltrainData = np.array(trainData[:90000])
    wordList = list()
    for words in finaldata:
        wordList.append(words)
    model = gensim.models.Word2Vec(wordList,min_count=1,size=200, window = 5, sg = 1) 
    word_vectors = model.wv
    #print model['Thanks']
    validData = np.array(trainData[90000:105000])
    validLabels = np.array(trainLabels[90000:105000])
    testData = np.array(trainData[105000:])
    testLabels = np.array(trainLabels[105000:])
    tf.reset_default_graph()
    with tf.Session() as session:
        W, b, X, predicted_outputs, Y, lamb1, optimizer, loss = buildGraph(2250*4,0.001,"CE")
        softmax1 = tf.nn.softmax(predicted_outputs,name = "Softmax")
        correct_pred = tf.equal(tf.argmax(softmax1, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.global_variables_initializer()
        session.run(init)
        start = time.time()
        train_dataset = tf.data.Dataset.from_tensor_slices(finaltrainData)
        test_dataset = tf.data.Dataset.from_tensor_slices(finaltrainLabels)
        valid_dataset = tf.data.Dataset.from_tensor_slices(validData)
        valid_labels_dataset = tf.data.Dataset.from_tensor_slices(validLabels)
        for epoch in range(100):
            print (epoch)    
            combindedDataset = tf.data.Dataset.zip((train_dataset, test_dataset)).shuffle(finaltrainLabels.shape[0]).batch(512)
            combindedDataset1 = tf.data.Dataset.zip((valid_dataset, valid_labels_dataset)).shuffle(validData.shape[0]).batch(1024)
            iterator = combindedDataset.make_initializable_iterator()
            iterator1 = combindedDataset1.make_initializable_iterator()
            next_element = iterator.get_next()
            session.run(iterator.initializer)
            numberOfBatches = int(finaltrainLabels.shape[0]/512)
            #print numberOfBatches
            for i in range(numberOfBatches):
                val = session.run(next_element)
                finalVectordata = list()
                for sentence in val[0]:
                    vectorData = list()
                    tokenizer = nltk.tokenize.WhitespaceTokenizer()
                    token = tokenizer.tokenize(sentence.decode("utf-8"))
                    for word in token:
                        vectorData.append(model[word])
                    while np.array(vectorData).shape[0] < maximum:
                        vectorData.append(np.zeros(200))
                    finalVectordata.append(np.concatenate(vectorData, axis=None))
                trainingdata = np.array(finalVectordata)
                trainingLabels = convertToOneHot(val[1])
                session.run(optimizer, feed_dict = {X:trainingdata,Y:trainingLabels,lamb1:reg})
            print ("Training Loss is " + str(session.run(loss,feed_dict={X: trainingdata,Y:trainingLabels,lamb1:reg})))
            next_element1 = iterator1.get_next()
            session.run(iterator1.initializer)
            val1 = session.run(next_element1)
            finalVectordata = list()
            for sentence in val1[0]:
                vectorData = list()
                tokenizer = nltk.tokenize.WhitespaceTokenizer()
                token = tokenizer.tokenize(sentence.decode("utf-8"))
                for word in token:
                    vectorData.append(model[word])
                while np.array(vectorData).shape[0] < maximum:
                    vectorData.append(np.zeros(200))
                finalVectordata.append(np.concatenate(vectorData, axis=None))
            validationData = np.array(finalVectordata)
            validationLabels = convertToOneHot(val1[1])
            session.run(optimizer, feed_dict = {X:validationData,Y:validationLabels,lamb1:reg})
            validationAccuracy = session.run(accuracy, feed_dict={X:validationData,Y:validationLabels,lamb1:reg})
            print("validation accuracy %g"%(validationAccuracy))
            validationError =  (session.run(loss,feed_dict={X:validationData ,Y:validationLabels,lamb1:reg}))
            print ("validation error %g"%validationError)
            iterationsList.append(epoch)
            validationLossList.append(validationError)
            validationAccuracyList.append(validationAccuracy)
            trainingAccuracy = session.run(accuracy, feed_dict={X:trainingdata,Y:trainingLabels,lamb1:reg})
            trainingLoss = session.run(loss, feed_dict={X:trainingdata,Y:trainingLabels,lamb1:reg})
            trainingLossList.append(trainingLoss)
            trainingAccuracyList.append(trainingAccuracy)
        print ("Training time is %g"%(time.time() - start))
        finalVectordata = list()
        for sentence in validData[:5000]:
            vectorData = list()
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            token = tokenizer.tokenize(sentence)
            for word in token:
                vectorData.append(model[word])
            while np.array(vectorData).shape[0] < maximum:
                vectorData.append(np.zeros(200))
            finalVectordata.append(np.concatenate(vectorData, axis=None))
        validationLabels = convertToOneHot(validLabels[:5000])
        validationAccuracy = session.run(accuracy, feed_dict={X:finalVectordata,Y:validationLabels,lamb1:reg})
        print("Final validation accuracy %g"%(validationAccuracy))
        validationError =  (session.run(loss,feed_dict={X: finalVectordata,Y:validationLabels,lamb1:reg}))
        print ("Final validation error %g"%validationError)
        finalVectordata = list()
        for sentence in testData[:5000]:
            vectorData = list()
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            token = tokenizer.tokenize(sentence)
            for word in token:
                vectorData.append(model[word])
            while np.array(vectorData).shape[0] < maximum:
                vectorData.append(np.zeros(200))
            finalVectordata.append(np.concatenate(vectorData, axis=None))
        testingData = np.array(finalVectordata)
        testingLabels = convertToOneHot(testLabels)
        testAccuracy = session.run(accuracy, feed_dict={X:finalVectordata,Y:testingLabels[:5000],lamb1:reg})
        print("Test accuracy %g"%(testAccuracy))
        testError =  (session.run(loss,feed_dict={X:finalVectordata,Y:testingLabels[:5000],lamb1:reg}))
        print ("Test error %g"%testError)
    plt.figure()
    plt.plot(iterationsList, trainingLossList, 'r')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(iterationsList, validationLossList, 'b')
    plt.gca().legend(('training Loss','validation Loss'))

    plt.savefig('error' + '.png')
    
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(iterationsList, trainingAccuracyList, 'r')
    plt.plot(iterationsList, validationAccuracyList, 'b')
    plt.gca().legend(('training Accuracy','validation Accuracy'))

    plt.savefig('accuracy' + '.png')
if __name__ == '__main__':
    main()

