import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


data = pd.read_csv("data.csv")

def part2():
    X = data.drop(['y'], axis=1, inplace=False)
    Y = data.loc[:,['y']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False)

    cValue = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]
    maeTrain_list = []
    maeValidation_list = []
    size = int(len(X_train)/10)
    maesum = 0

    #CV with 10 folds for each C value
    for c in cValue:
        start = 0
        maeTrain = 0
        maeValidation = 0
        mae = []
        for k in range(10):
            CVx = X_train[start:(k+1)*size]
            CVy = y_train[start:(k+1)*size]

            trainX = pd.concat([X_train[:start], X_train[(k+1)*size:]])
            trainY = pd.concat([y_train[:start], y_train[(k+1)*size:]])
            start += size

            #fit the data with SVC
            clf = SVC(C=c, kernel = 'linear', gamma='auto')
            clf.fit(trainX, trainY)

            CVPredict = clf.predict(CVx)#.reshape(-1,1)

            #Find the training and validation score
            maeValidation = mean_absolute_error(CVy,CVPredict)
            mae.append(maeValidation)

        print("CValue = {}, CV error rate: {}".format(c, np.mean(mae)))

        maeValidation_list.append(np.mean(mae))

    print("C Value 10 gave the lowest error rate of 0.13333. Therefore it is the best C value for this dataset")
    #plot graph
    plt.plot(cValue, maeValidation_list, label= 'Validation Data')
    plt.xlabel('C Value')
    plt.ylabel('Error Rate')
    plt.xscale('log')
    plt.legend()
    plt.title('C Value Graph')
    plt.show()

    clf = SVC(C=10, kernel = 'linear', gamma='auto')
    clf.fit(X_train, y_train)

    testPredict = clf.predict(X_test)
    bestCMae = mean_absolute_error(y_test,testPredict)

    print("C 10 test error is {}".format(bestCMae))
    red = np.array(X_test)[np.where(y_test == 0)[0], :]
    blue = np.array(X_test)[np.where(y_test == 1)[0], :]

    plt.scatter(red[:, 0], red[:, 1], c='r', label="0")
    plt.scatter(blue[:, 0], blue[:, 1], c='b', label="1")
    plot_svc_decision_function(clf)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Scatter graph for C Value 10')
    plt.show()


def number9():
    X = data.drop(['y'], axis=1, inplace=False)
    Y = data.loc[:,['y']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False)

    cValue = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]
    maeTrain_list = []
    maeValidation_list = []
    size = int(len(X_train)/10)
    maesum = 0

    #CV with 10 folds for each C value
    for c in cValue:
        start = 0
        maeTrain = 0
        maeValidation = 0
        mae = []
        for k in range(10):
            CVx = X_train[start:(k+1)*size]
            CVy = y_train[start:(k+1)*size]

            trainX = pd.concat([X_train[:start], X_train[(k+1)*size:]])
            trainY = pd.concat([y_train[:start], y_train[(k+1)*size:]])
            start += size

            #fit the data with SVC
            clf = SVC(C=c, kernel = 'linear', gamma='auto')
            clf.fit(trainX, trainY)

            CVPredict = clf.predict(CVx)#.reshape(-1,1)

            #Find the training and validation score
            maeValidation = mean_absolute_error(CVy,CVPredict)
            mae.append(maeValidation)

        print("CValue = {}, CV error rate: {}".format(c, np.mean(mae)))

        maeValidation_list.append(np.mean(mae))

    print("C Value 10 gave the lowest error rate of 0.13333. Therefore it is the best C value for this dataset")
    #plot graph
    plt.plot(cValue, maeValidation_list, label= 'Validation Data')
    plt.xlabel('C Value')
    plt.ylabel('Error Rate')
    plt.xscale('log')
    plt.legend()
    plt.title('C Value Graph')
    plt.show()

    clf = SVC(C=10, kernel = 'rbf', gamma='auto')
    clf.fit(X_train, y_train)

    testPredict = clf.predict(X_test)
    bestCMae = mean_absolute_error(y_test,testPredict)

    print("C 10 test error is {}".format(bestCMae))
    red = np.array(X_test)[np.where(y_test == 0)[0], :]
    blue = np.array(X_test)[np.where(y_test == 1)[0], :]

    plt.scatter(red[:, 0], red[:, 1], c='r', label="0")
    plt.scatter(blue[:, 0], blue[:, 1], c='b', label="1")
    plot_svc_decision_function(clf)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Scatter graph for C Value 10')
    plt.show()

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


part2()
number9()
# Number 9)
