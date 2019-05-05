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
from glove import Corpus, Glove


validLabels = list()
validData = list()
polarity = {'empty' : 'N','sadness' : 'N','enthusiasm' : 'P','neutral' : 'neutral','worry' : 'N','surprise' : 'P','love' : 'P','fun' : 'P','hate' : 'N','happiness' : 'P','boredom' : 'N','relief' : 'P','anger' : 'N'}  
labels = {'N': 0, 'P':1, 'neutral':2}
reg = 0.1
def convertToOneHot(trainLabels):
    oneHottrainLabels = np.zeros((trainLabels.shape[0],4))
    for item,data in enumerate(trainLabels):
        oneHottrainLabels[item][int(data) - 1] = 1 
    return oneHottrainLabels

def countWords(sentence):
	count = 0
	tokenizer = nltk.tokenize.WhitespaceTokenizer()
	tokens = tokenizer.tokenize(sentence)
	for token in tokens:
		if (size(token)) > 1:
			count+1
	return count

def createFreqDict(finalTrainData):
    freqDictList = list()
    i = 0
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    for sentence in finalTrainData:
        i += 1
        freq_dict = {}
        words = tokenizer.tokenize(sentence)
        if (len(words) >= 1):
            for word in words:
                if word in freq_dict:
                    freq_dict[word] += 1
                else: 
                    freq_dict[word] = 1
            temp_dict = {'doc_id':i, 'freq_dict':freq_dict}
            freqDictList.append(temp_dict)
    return freqDictList

def createDocDict(finalTrainData):
    freqDictList = list()
    i = 0
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    for data in finalTrainData:
        i += 1
        temp_dict = {'doc_id':i, 'doc_length':len(tokenizer.tokenize(data))}
        freqDictList.append(temp_dict)
    return freqDictList

def computeTF(doc_info, freq_dict_list):
	TF_scores = []
	for temp_dict in freq_dict_list:
		id = temp_dict['doc_id']
		for k in temp_dict['freq_dict']:
			temp = { 'doc_id' : id, 'TF_score' : float(temp_dict['freq_dict'][k]/doc_info[id-1]['doc_length']), 'key':k}
			TF_scores.append(temp)
	return TF_scores

def computeIDF(doc_info, freqDictList):
    IDF_Scores = []
    counter = 0
    for dict1 in freqDictList:
        counter +=1
        for k in dict1['freq_dict'].keys():
            count = sum([k in temp_dict['freq_dict'] for temp_dict in freqDictList ])
            temp = {'doc_id': counter, 'IDF_score': math.log(len(doc_info)/count), 'key' : k}
            IDF_Scores.append(temp)
    return IDF_Scores

def computeIDFTF(TF_scores, IDF_Scores):
    TFIDF_Scores = []
    count = 0;
    for j in IDF_Scores:
        for i in TF_scores:
            if j['key'] == i['key'] and j['doc_id'] == i['doc_id']:
                count = count + 1
                temp = {'doc_id': j['doc_id'],'TFIDF_score':j['IDF_score']*i['TF_score'],'key':i['key']}
        if (count > 1):
            TFIDF_Scores.append(temp)
            count = 0
    return TFIDF_Scores

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

def measure_Accuracy(new_W, new_b, testData, testLabels):
    print new_W.shape
    print new_b.shape
    print testData.shape
    print testLabels.shape
    Y_prediction = new_W.T.dot(testData.T) + new_b
    for i in range(Y_prediction.shape[0]):
        maximumpredicition = max(Y_prediction[i][0], Y_prediction[i][1], Y_prediction[i][2], Y_prediction[i][3])
        maximumLabel = max(testLabels[i][0],testLabels[i][1], testLabels[i][2], testLabels[i][3])
        if maximumpredicition == Y_prediction[i][0]:
            labelprediction = 0
        elif maximumpredicition == Y_prediction[i][1]:
            labelprediction = 1
        elif maximumpredicition == Y_prediction[i][2]:
            labelprediction = 2
        elif maximumpredicition == Y_prediction[i][3]:
            labelprediction = 3
        if maximumLabel == testLabels[i][0]:
            label = 0
        elif maximumLabel == testLabels[i][1]:
            label = 1
        elif maximumLabel == testLabels[i][2]:
            label = 2
        elif maximumLabel == testLabels[i][3]:
            label = 3

        if Y_prediction[i][0] >= 0.5:
            Y_prediction[i][0] = 1
        else:
            Y_prediction[i][0] = 0
        if Y_prediction[i][1] >= 0.5:
            Y_prediction[i][1] = 1
        else:
            Y_prediction[i][1] = 0
        if Y_prediction[i][2] >= 0.5:
            Y_prediction[i][2] = 1
        else:
            Y_prediction[i][2] = 0
        if Y_prediction[i][3] >= 0.5:
            Y_prediction[i][3] = 1
        else:
            Y_prediction[i][3] = 0

    print Y_prediction.shape
    accuracy = np.mean(abs(Y_prediction.T - testLabels))
    print 100*(1-accuracy)
    totalcount += 1
    if label == labelprediction:
        counter+=1;
    accuracy1 = counter/totalcount
    print 100*(accuracy1)
    return 100*(1-accuracy)

def stochastic_gradient_descent(maximum, batchSize, lamb, learningRate,epochs,data,oneHotLabels, filename, beta1, beta2, epsilon, lossType):
    tf.reset_default_graph()
    trainoneHotLabels = np.array(oneHotLabels[:28000,:])
    print trainoneHotLabels.shape
    validoneHotLabels = np.array(oneHotLabels[28000:35000,:])
    print validoneHotLabels.shape
    testoneHotLabels = np.array(oneHotLabels[35000:40000,:])
    print testoneHotLabels.shape
    with tf.Session() as session:
        outputFilename = filename + "__beta2__" + str(beta2) + "_epochs_" + str(epochs) +  "_LossType_" +  str(lossType) + "_LearningRate_" + str(learningRate) + "_BatchSize_" + str(batchSize) + "_Lambda_" + str(lamb) + ".csv";
        writer = tf.summary.FileWriter('./graphs/' + outputFilename,session.graph)
        W, b, traindata, predicted_outputs, trainlabels, lamb1, optimizer, loss = buildGraph(maximum,beta1,beta2,epsilon,lossType,learningRate)
        init = tf.global_variables_initializer()
        session.run(init)
        trainingLossList = list();
        testLossList = list()
        validationLossList = list()
        iterationsList = list()
        validationAccuracyList = list();
        testAccuracyList = list()
        start = time.time()
        summaryTensor = tf.summary.scalar(name = "error",tensor = loss)
        validData = compute(data[28000:35000],maximum)
        print validData.shape
        testData = compute(data[35000:35100],maximum)
        print testData.shape
        for epoch in range(epochs):
            train_dataset = tf.data.Dataset.from_tensor_slices(data[:28000])
            test_dataset = tf.data.Dataset.from_tensor_slices(trainoneHotLabels)
            combindedDataset = tf.data.Dataset.zip((train_dataset, test_dataset)).shuffle(trainoneHotLabels.size).batch(batchSize)
            iterator = combindedDataset.make_initializable_iterator()
            # extract an element
            next_element = iterator.get_next()
            session.run(iterator.initializer)
            numberOfBatches = int(trainoneHotLabels.shape[0]/batchSize)
            for i in range(numberOfBatches):
                val = session.run(next_element)
                trainingdata = compute(val[0],maximum)
                session.run(optimizer, feed_dict = {traindata:trainingdata,trainlabels:val[1],lamb1:lamb})
                print "Training Loss is " + str(session.run(loss,feed_dict={traindata: trainingdata,trainlabels:val[1],lamb1:lamb}))
            print " Validation Loss is" + str(session.run(loss,feed_dict={traindata: validData,trainlabels:validoneHotLabels,lamb1:lamb}))
        print " Test accuracy is" + str(measure_Accuracy(session.run(W),session.run(b),testData, testoneHotLabels))

def preprocessData(trainData):
    stop_words = set(stopwords.words('english')) 
    sentence = list()
    finalTrainData = list()
    l = 0
    for data in trainData:
        updatedText = BeautifulSoup(data, 'lxml')
        updatedText = re.sub(r'@[A-Za-z0-9]+','',updatedText.get_text())
        updatedText = re.sub(r'=','',updatedText)
        updatedText = re.sub('https?://[A-Za-z0-9./]+','',updatedText)
        #updatedText = updatedText.decode("utf-8-sig")
        #updatedText = updatedText.replace(u"\ufffd", "?")
        letters_only = re.sub("[^a-zA-Z]", " ", updatedText)
        lower_case = letters_only.lower()
        tokenizer = nltk.tokenize.WordPunctTokenizer()
        tokens = tokenizer.tokenize(lower_case)
        stemmer = nltk.stem.WordNetLemmatizer()
        for token in tokens:
            if token not in stop_words: 
                sentence.append(stemmer.lemmatize(token))
                #print sentence
        sentence1 = (" ".join(sentence)).strip()
        finalTrainData.append(sentence1)
        if l < len(tokenizer.tokenize(sentence1)):
            l = len(tokenizer.tokenize(sentence1))
            print l
            print sentence1
        sentence = []

    print np.array(finalTrainData).shape
    return np.array(finalTrainData), l

def compute(finalTrainData,maximum):
    vectorData = list()
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    freqDictList = createFreqDict(finalTrainData)
    docDictList = createDocDict(finalTrainData)
    tfScores = computeTF(docDictList, freqDictList)
    IDFScores = computeIDF(docDictList, freqDictList)
    TFIDFScores = computeIDFTF(tfScores, IDFScores)
    tfidfscoresList = list()
    wordList = list()
    for item in TFIDFScores:
        tfidfscoresList.append(item['TFIDF_score'])
        wordList.append(item['key'])
    for sentence in finalTrainData:
            vector = np.zeros(maximum)
            words = tokenizer.tokenize(sentence)
            for i,word in enumerate(words):
                for k,item in enumerate(wordList):
                    if word == item:
                        vector[i] = tfidfscoresList[k]
                        break;
            vectorData.append(vector)  
    return np.array(vectorData)

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
                    print sentence1
    print maximum           
    f.close()
    print len(trainLabels)
    finaltrainLabels = np.array(trainLabels[:90000])
    finaltrainData = np.array(trainData[:90000])
    corpus = Corpus() 
    #training the corpus to generate the co occurence matrix which is used in GloVe
    corpus.fit(finaldata, window=10)
    #creating a Glove object which will use the matrix created in the above lines to create embeddings
    #We can set the learning rate as it uses Gradient Descent and number of components
    glove = Glove(no_components=50, learning_rate=0.05)
     
    glove.fit(corpus.matrix, epochs=100, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')
    glove.add_dictionary(corpus.dictionary)

    #model = gensim.models.Word2Vec(wordList,min_count=1,size=200, window = 5, sg = 0) 
    #word_vectors = model.wv
    #print model['Thanks']
    print glove.word_vectors[glove.dictionary['thanks']]
    validData = np.array(trainData[90000:105000])
    validLabels = np.array(trainLabels[90000:105000])
    testData = np.array(trainData[105000:])
    testLabels = np.array(trainLabels[105000:])
    tf.reset_default_graph()
    with tf.Session() as session:
        W, b, X, predicted_outputs, Y, lamb1, optimizer, loss = buildGraph(2250,0.0001,"CE")
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
            print epoch    
            combindedDataset = tf.data.Dataset.zip((train_dataset, test_dataset)).shuffle(finaltrainLabels.shape[0]).batch(500)
            combindedDataset1 = tf.data.Dataset.zip((valid_dataset, valid_labels_dataset)).shuffle(validData.shape[0]).batch(1000)
            iterator = combindedDataset.make_initializable_iterator()
            iterator1 = combindedDataset1.make_initializable_iterator()
            next_element = iterator.get_next()
            session.run(iterator.initializer)
            numberOfBatches = int(finaltrainLabels.shape[0]/500)
            print numberOfBatches
            for i in range(numberOfBatches):
                val = session.run(next_element)
                finalVectordata = list()
                for sentence in val[0]:
                    vectorData = list()
                    tokenizer = nltk.tokenize.WhitespaceTokenizer()
                    token = tokenizer.tokenize(sentence)
                    for word in token:
                        vectorData.append(glove.word_vectors[glove.dictionary[word]])
                    while np.array(vectorData).shape[0] < maximum:
                        vectorData.append(np.zeros(50))
                    finalVectordata.append(np.concatenate(vectorData, axis=None))
                trainingdata = np.array(finalVectordata)
                trainingLabels = convertToOneHot(val[1])
                session.run(optimizer, feed_dict = {X:trainingdata,Y:trainingLabels,lamb1:reg})
            print "Training Loss is " + str(session.run(loss,feed_dict={X: trainingdata,Y:trainingLabels,lamb1:reg}))
            next_element1 = iterator1.get_next()
            session.run(iterator1.initializer)
            val1 = session.run(next_element1)
            finalVectordata = list()
            for sentence in val1[0]:
                vectorData = list()
                tokenizer = nltk.tokenize.WhitespaceTokenizer()
                token = tokenizer.tokenize(sentence)
                for word in token:
                    vectorData.append(glove.word_vectors[glove.dictionary[word]])
                while np.array(vectorData).shape[0] < maximum:
                    vectorData.append(np.zeros(50))
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
        finalVectordata = list()
        for sentence in validData:
            vectorData = list()
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            token = tokenizer.tokenize(sentence)
            for word in token:
                vectorData.append(glove.word_vectors[glove.dictionary[word]])
            while np.array(vectorData).shape[0] < maximum:
                vectorData.append(np.zeros(50))
            finalVectordata.append(np.concatenate(vectorData, axis=None))
        validationAccuracy = session.run(accuracy, feed_dict={X:validationData,Y:validationLabels,lamb1:reg})
        print("Final validation accuracy %g"%(validationAccuracy))
        validationError =  (session.run(loss,feed_dict={X:validationData ,Y:validationLabels,lamb1:reg}))
        print ("Final validation error %g"%validationError)
        finalVectordata = list()
        for sentence in testData:
            vectorData = list()
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            token = tokenizer.tokenize(sentence)
            for word in token:
                vectorData.append(glove.word_vectors[glove.dictionary[word]])
            while np.array(vectorData).shape[0] < maximum:
                vectorData.append(np.zeros(50))
            finalVectordata.append(np.concatenate(vectorData, axis=None))
        testingData = np.array(finalVectordata)
        testingLabels = convertToOneHot(testLabels)
        testAccuracy = session.run(accuracy, feed_dict={X:testingData,Y:testingLabels,lamb1:reg})
        print("Test accuracy %g"%(testAccuracy))
        testError =  (session.run(loss,feed_dict={X:testingData ,Y:testingLabels,lamb1:reg}))
        print ("Test error %g"%testError)
    plt.figure()
    plt.plot(iterationsList, trainingLossList, 'r')
    plt.plot(iterationsList, validationLossList, 'b')
    plt.gca().legend(('training Loss','validation Loss'))

    plt.savefig('Size-200-reg0.1-batch-500_error_' + '.png')
    
    plt.figure()
    plt.plot(iterationsList, trainingAccuracyList, 'r')
    plt.plot(iterationsList, validationAccuracyList, 'b')
    plt.gca().legend(('training Accuracy','validation Accuracy'))

    plt.savefig('Size-200_reg0.1-batch-500_accuracy_' + '.png')
if __name__ == '__main__':
    main()

