import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Ridge,Lasso
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import loaddata_lab5_Huang_Tony as loader

data = loader.load("data_lab5.csv")

scaler = preprocessing.StandardScaler()

#Splitting the training and testing data
train_ratio = 0.3
num_rows = data.shape[0]
train_set_size = int(num_rows * train_ratio)

train_data = data.iloc[:train_set_size]
test_data = data.iloc[train_set_size:]

train_features = train_data.drop(['TARGET_D'], axis=1, inplace=False)
standardize_train_features = scaler.fit_transform(train_features)
train_features = pd.DataFrame(standardize_train_features)
train_labels = train_data.loc[:,['TARGET_D']]

test_features = test_data.drop(['TARGET_D'], axis=1, inplace=False)
standardize_test_features = scaler.fit_transform(test_features)
test_features = pd.DataFrame(standardize_test_features)
test_labels = test_data.loc[:,['TARGET_D']]

kFoldsX = train_features
kFoldsY = train_labels


maeTrain_list = []
maeValidation_list = []
size = int(len(train_features)/5)
lambdaValues = range(-3,11)

#CV with 5 folds for each lambda value
for l in lambdaValues:
    start = 0
    maeTrain = 0
    maeValidation = 0
    for k in range(5):
        CVx = kFoldsX[start:(k+1)*size]
        CVy = kFoldsY[start:(k+1)*size]
        trainX = pd.concat([kFoldsX[:start], kFoldsX[(k+1)*size:]])
        trainY = pd.concat([kFoldsY[:start], kFoldsY[(k+1)*size:]])
        start += size

        trainRidge = Ridge(alpha=(10 ** l))
        trainRidge.fit(trainX, trainY)
        CVPredict = trainRidge.predict(CVx)
        trainPredict = trainRidge.predict(trainX)

        maeTrain += float(np.mean(abs(trainY - trainPredict)))
        maeValidation += float(np.mean(abs(CVy - CVPredict)))
    print("lambda = 10^{}, TRAIN MAE: {}".format(l, maeTrain/5))
    print("lambda = 10^{}, CV MAE: {}".format(l, maeValidation/5))
    maeTrain_list.append(maeTrain/5)
    maeValidation_list.append(maeValidation/5)

#plot graph
plt.plot(lambdaValues, maeTrain_list, label ='MAE Training Data')
plt.plot(lambdaValues, maeValidation_list, label= 'Validation Data')
plt.xlabel('Lambda')
plt.ylabel('MAE')
plt.legend()
plt.title('Ridge Regression Graph')
plt.show()

print("The best value for lambda is 10^4")
testRidge = Ridge(alpha=(10 ** 4))
testRidge.fit(train_features, train_labels)
testPredict = testRidge.predict(test_features)
newMAE = float(np.mean(abs(test_labels - testPredict)))
print("The MAE value with lambda 10^4 is: {}".format(newMAE))
print("The MAE value is significantly decreased compared to the first part.")
