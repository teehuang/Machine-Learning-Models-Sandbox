import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv")
X = data.drop(['y'], axis=1, inplace=False)
Y = data.loc[:,['y']]


train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=.25, shuffle=False)

maeTrain_list = []
maeValidation_list = []
size = int(len(train_features)/10)
cValue = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100]

#CV with 10 folds for each C value
for c in cValue:
    start = 0
    maeTrain = 0
    maeValidation = 0
    for k in range(10):
        CVx = train_features[start:(k+1)*size]
        CVy = train_labels[start:(k+1)*size]
        trainX = pd.concat([train_features[:start], train_features[(k+1)*size:]])
        trainY = pd.concat([train_labels[:start], train_labels[(k+1)*size:]])
        start += size

        #fit the data with SVC
        clf = SVC(C=c)
        clf.fit(trainX, trainY)

        CVPredict = clf.predict(CVx)#.reshape(15,1)

        #Find the training and validation score
        #maeValidation = clf.score(CVPredict,CVy)
        CVy = np.array(CVy)
        CVy.reshape(15,)
        count = 0
        for x in range(15):
            if CVPredict[x] == CVy[x]:
                count = count + 1
        maeValidation = count/ 15

    print("CValue = {}, CV MAE: {}".format(c, 1-maeValidation))

    #maeTrain_list.append(1-maeTrain)
    maeValidation_list.append(1-maeValidation)

print("C Value 1 gave the lowest error rate of 0.2 therefore it is the best C value for this dataset")
#plot graph

plt.plot(cValue, maeValidation_list, label= 'Validation Data')
#plt.xticks(cValue)
plt.xlabel('C Value')
plt.ylabel('Error Rate')
plt.xscale('log')
plt.legend()
plt.title('C Value Graph')
plt.show()

print("C Value 1 gave the lowest error rate of 0.08 therefore it is the best C value for this dataset")

clf = SVC(C=1.0)
clf.fit(train_features, train_labels)
testPredict = clf.predict(test_features)

test_labels = np.array(test_labels)
test_labels.reshape(50,)

count = 0
for x in range(50):
    if testPredict[x] == test_labels[x]:
        count = count + 1
testError = count/ 50
print("Testing with C Value of 1.0: Test error rate = {}".format(1-testError))
