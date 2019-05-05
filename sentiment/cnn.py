import csv
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from nltk.data import find
import gensim
from gensim.models import Word2Vec
trainingData = list();
trainingLabels = list()
validationData = list()
validationLabels = list()
import time
learning_rate = 0.001


def convertToOneHot(trainLabels):
    oneHottrainLabels = np.zeros((trainLabels.shape[0],4))
    for item,data in enumerate(trainLabels):
        oneHottrainLabels[item][int(data) - 1] = 1 
    return oneHottrainLabels

def conv2d(x, W, b, name,strides=2):
    x = tf.nn.conv1d(value = x, filters = W, stride=strides, padding='SAME', name = name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x,name,k=2):
    return tf.nn.pool(x, [5], 'AVG', 'SAME', strides = [k])
    #return tf.nn.max_pool(value = x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME', name = name)

def conv_net(x, weights, biases):
    x = tf.reshape(x, shape=[-1, 2250,1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'],"Convolution1")
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.dropout(conv1,0.2)
    conv1 = maxpool2d(conv1,"Pooling2",5)
    #print conv1.shape
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],"Convolution2")
    conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.dropout(conv2,0.2)
    conv2 = maxpool2d(conv2,"Pooling2",5)
    print (conv2.shape)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'],"Convolution3")
    conv3 = tf.layers.batch_normalization(conv3)
    conv3 = tf.nn.dropout(conv3,0.2)
    conv3 = maxpool2d(conv3,"Pooling2",5)
    print (conv3.shape)
    fc1 = tf.contrib.layers.flatten(conv3)
    print (fc1.shape)
    fc1 = tf.matmul(fc1, weights['wd1']) +  biases['bd1']
    fc1 = tf.nn.relu(fc1,name ='finalrelu')
    print (fc1.shape)
    #fc1 = tf.nn.dropout(fc1, 0.5,name = 'Dropout')
    logits = tf.matmul(fc1, weights['out']) + biases['outb']
    print (logits.shape)
    return logits
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
    model = gensim.models.Word2Vec(wordList,min_count=1,size=50, window = 5, sg = 1) 
    word_vectors = model.wv
    #print model['Thanks']
    validData = np.array(trainData[90000:105000])
    validLabels = np.array(trainLabels[90000:105000])
    testData = np.array(trainData[105000:])
    testLabels = np.array(trainLabels[105000:])
    tf.reset_default_graph()
    session = tf.Session()
    initializer = tf.contrib.layers.xavier_initializer()
    with session.as_default():
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.get_variable("wc1",shape = [5, 1, 128], initializer = tf.contrib.layers.xavier_initializer()),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.get_variable( "wc2", shape = [5, 128, 128], initializer = tf.contrib.layers.xavier_initializer()),

            'wc3': tf.get_variable( "wc3", shape = [5, 128, 128], initializer = tf.contrib.layers.xavier_initializer()),

            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.get_variable("wd1", shape = [384, 512], initializer = tf.contrib.layers.xavier_initializer()),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.get_variable("out",shape = [512, 4], initializer = tf.contrib.layers.xavier_initializer())
            }

        biases = {
            'bc1': tf.get_variable("bc1",shape = [128], initializer = tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable("bc2", shape = [128], initializer = tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable("bc3", shape = [128],initializer = tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable("bd1",shape = [512],initializer = tf.contrib.layers.xavier_initializer()),
            'outb': tf.get_variable("outb", shape = [4],initializer = tf.contrib.layers.xavier_initializer())
        }
        
        X = tf.placeholder(tf.float32, [None, 2250,1], name='input')
        Y = tf.placeholder(tf.float32, [None, 4])
        logits = conv_net(X, weights, biases)
        print (logits.shape)
        softmax1 = tf.nn.softmax(logits,name = "Softmax")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        #reg = tf.multiply(0.1 / 2, tf.reduce_sum(tf.square(weights['out'])))
        totalLoss = loss 
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(totalLoss)
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
            #print (numberOfBatches)
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
                        vectorData.append(np.zeros(50))
                    finalVectordata.append(np.concatenate(vectorData, axis=None))
                trainingdata = np.array(finalVectordata)
                trainingdata = np.expand_dims(trainingdata, axis=2)
                trainingLabels = convertToOneHot(val[1])
                session.run(optimizer, feed_dict = {X:trainingdata,Y:trainingLabels})
            print ("Training Loss is " + str(session.run(loss,feed_dict={X: trainingdata,Y:trainingLabels})))
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
                    vectorData.append(np.zeros(50))
                finalVectordata.append(np.concatenate(vectorData, axis=None))
            validationData = np.array(finalVectordata)
            validationData = np.expand_dims(validationData, axis=2)

            validationLabels = convertToOneHot(val1[1])
            session.run(optimizer, feed_dict = {X:validationData,Y:validationLabels})
            validationAccuracy = session.run(accuracy, feed_dict={X:validationData,Y:validationLabels})
            print("validation accuracy %g"%(validationAccuracy))
            validationError =  (session.run(loss,feed_dict={X:validationData ,Y:validationLabels}))
            print ("validation error %g"%validationError)
            iterationsList.append(epoch)
            validationLossList.append(validationError)
            validationAccuracyList.append(validationAccuracy)
            trainingAccuracy = session.run(accuracy, feed_dict={X:trainingdata,Y:trainingLabels})
            trainingLoss = session.run(loss, feed_dict={X:trainingdata,Y:trainingLabels})
            trainingLossList.append(trainingLoss)
            trainingAccuracyList.append(trainingAccuracy)
        print ("Training Time is %g"%(time.time() - start))
        finalVectordata = list()
        for sentence in validData[:2000]:
            vectorData = list()
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            token = tokenizer.tokenize(sentence)
            for word in token:
                vectorData.append(model[word])
            while np.array(vectorData).shape[0] < maximum:
                vectorData.append(np.zeros(50))
            finalVectordata.append(np.concatenate(vectorData, axis=None))
        validationData = np.array(finalVectordata)
        validationData = np.expand_dims(validationData, axis=2)
        validationLabels = convertToOneHot(validLabels[:2000])
        validationAccuracy = session.run(accuracy, feed_dict={X:validationData,Y:validationLabels})
        print("Final validation accuracy %g"%(validationAccuracy))
        validationError =  (session.run(loss,feed_dict={X:validationData ,Y:validationLabels}))
        print ("Final validation error %g"%validationError)
        finalVectordata = list()
        for sentence in testData[:2000]:
            vectorData = list()
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            token = tokenizer.tokenize(sentence)
            for word in token:
                vectorData.append(model[word])
            while np.array(vectorData).shape[0] < maximum:
                vectorData.append(np.zeros(50))
            finalVectordata.append(np.concatenate(vectorData, axis=None))
        testingData = np.array(finalVectordata)
        testingData = np.expand_dims(testingData, axis=2)
        testingLabels = convertToOneHot(testLabels[:2000])
        testAccuracy = session.run(accuracy, feed_dict={X:testingData,Y:testingLabels})
        print("Test accuracy %g"%(testAccuracy))
        testError =  (session.run(loss,feed_dict={X:testingData ,Y:testingLabels}))
        print ("Test error %g"%testError)
    plt.figure()
    plt.plot(iterationsList, trainingLossList, 'r')
    plt.plot(iterationsList, validationLossList, 'b')
    plt.gca().legend(('training Loss','validation Loss'))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('error' + '.png')
    
    plt.figure()
    plt.plot(iterationsList, trainingAccuracyList, 'r')
    plt.plot(iterationsList, validationAccuracyList, 'b')
    plt.gca().legend(('training Accuracy','validation Accuracy'))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig('accuracy' + '.png')

if __name__ == '__main__':
    main()




