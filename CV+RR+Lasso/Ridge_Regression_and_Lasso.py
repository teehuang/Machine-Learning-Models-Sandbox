import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Ridge,Lasso
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import one_hot_encoding as loader

data = loader.load("data.csv")

def part1():
    train_ratio = 0.3
    num_rows = data.shape[0]
    train_set_size = int(num_rows * train_ratio)

    #Splitting the training and testing data
    train_data = data.iloc[:train_set_size]
    test_data = data.iloc[train_set_size:]
    train_features = train_data.drop(['TARGET_D'], axis=1, inplace=False)
    train_labels = train_data.loc[:,['TARGET_D'] ]
    test_features = test_data.drop(['TARGET_D'], axis=1, inplace=False)
    test_labels = test_data.loc[:,['TARGET_D']]

    lin_reg = LinearRegression()
    lin_reg.fit(train_features, train_labels)

    price_pred = lin_reg.predict(test_features)

    mae = np.mean(abs(test_labels - price_pred))
    print('Mean Absolute Error = ', mae)

def part2():
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



def part3():
    scaler = preprocessing.StandardScaler()

    #splitting the training and testing data
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
    lambdaValues = range(-8,9)

    #CV of 5 folds for each lambda value.
    for l in lambdaValues:
        start = 0
        maeTrain = 0
        maeValidation = 0
        for k in range(5):
            CVx = kFoldsX[start:(k+1)*size].values
            CVy = kFoldsY[start:(k+1)*size].values
            trainX = pd.concat([kFoldsX[:start], kFoldsX[(k+1)*size:]]).values
            trainY = pd.concat([kFoldsY[:start], kFoldsY[(k+1)*size:]]).values
            start += size

            trainLasso = Lasso(alpha=(10 ** (l/4.0)))
            trainLasso.fit(trainX, trainY)
            CVPredict = trainLasso.predict(CVx)
            trainPredict = trainLasso.predict(trainX)

            pdTrainY = pd.DataFrame(trainY)
            pdTrainPredict = pd.DataFrame(trainPredict)
            maeTrain += float(np.mean(abs(pdTrainY - pdTrainPredict)))

            pdCVy= pd.DataFrame(CVy)
            pdCVPredict = pd.DataFrame(CVPredict)
            maeValidation += float(np.mean(abs(pdCVy - pdCVPredict)))

        print("lambda = 10^{}, TRAIN MAE: {}".format((l/4.0), maeTrain/5))
        print("lambda = 10^{}, CV MAE: {}".format((l/4.0), maeValidation/5))
        maeTrain_list.append(maeTrain/5)
        maeValidation_list.append(maeValidation/5)

    lambdaManual = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    plt.plot(lambdaManual, maeTrain_list, label ='MAE Training Data')
    plt.plot(lambdaManual, maeValidation_list, label= 'Validation Data')
    plt.xlabel('Lambda')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Lasso Graph')
    plt.show()

    print("The best value for lambda is 10^-0.25")
    testLasso = Lasso(alpha=(10 ** -0.25))
    testLasso.fit(train_features.values, train_labels.values)
    testPredict = testLasso.predict(test_features.values)
    top3 = []



    newMAE = float(np.mean(abs(test_labels.values - testPredict)))
    print("The MAE value with lambda 10^-0.25 is: {}".format(newMAE))
    print("The MAE value obtained from Lasso vs Linear regression was incomparable. Lasso's MAE was significantly smaller")
    print("The MAE values with Lasso and Ridge Regression were close but in this case, the Ridge Regression had a smaller MAE.")
    print("In this case Ridge Regression is the best model to use because it had the lowest MAE value out of the 3 techniques we used.")

    # Find the top 3 coefficients
    findCoefficient = testLasso.coef_.tolist()
    for i in range(3):
        top3.append(max(findCoefficient))
        findCoefficient.remove(max(findCoefficient))

    # Find the features corresponding to the coefficient
    findIndex = testLasso.coef_.tolist()
    indeces = []
    count = 0
    for i in findIndex:
        for j in top3:
            if(j == i):
                indeces.append(count)
        count = count + 1


    print("Feature indeces: {}".format(indeces))
    listFeatures = list(data)

    print("The top 3 features are: {} {} {}".format(listFeatures[indeces[0]], listFeatures[indeces[1]], listFeatures[indeces[2]]))

#part1()
#part2()
part3()
