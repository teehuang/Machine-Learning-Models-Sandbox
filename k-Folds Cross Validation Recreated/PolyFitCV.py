import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def poly_kfoldCV(x,y,p,K):
    kFoldsX = x
    kFoldsY = y
    maeTrain = 0
    maeValidation = 0
    start = 0
    size = int(len(x)/K)
    for k in    range(K):
        foldx = kFoldsX[start:(k+1)*size]
        foldy = kFoldsY[start:(k+1)*size]
        leftX = pd.concat([kFoldsX[:start], kFoldsX[(k+1)*size:]])
        leftY = pd.concat([kFoldsY[:start], kFoldsY[(k+1)*size:]])

        start += size

        fit = np.polyfit(leftX,leftY, p)
        predictValidation = np.polyval(fit,foldx)
        predictTrain = np.polyval(fit,leftX)


        maeTrain += np.mean(abs(leftY - predictTrain))
        maeValidation += np.mean(abs(foldy - predictValidation))


    return maeTrain/K, maeValidation/K

def part2():
    trainError = []
    cvError = []
    for p in range(15):
        maeTrain, maeVal = poly_kfoldCV(x,y,p+1,5)
        trainError.append(maeTrain)
        cvError.append(maeVal)

    plt.plot(range(1, 16), trainError, label ='Train Error')
    plt.plot(range(1, 16), cvError, label= 'CV Error')
    plt.xlabel('p value')
    plt.ylabel('Prediction Error')
    plt.legend()
    plt.title('CV Graph Part 2')
    plt.show()

    print("The 5th degree polynomial held the least CV errors")

def part3():
    nList = range(20,100,5)
    pList = [1,2,7,10,16]
    for p in pList:
        trainError = []
        cvError = []
        for n in nList:
            maeTrain, maeVal = poly_kfoldCV(x[:n], y[:n],p,5)
            trainError.append(maeTrain)
            cvError.append(maeVal)
        plt.plot(nList, trainError, label ='Train Error')
        plt.plot(nList, cvError, label= 'CV Error')
        plt.ylim((0,2))
        plt.xlabel('p value')
        plt.ylabel('Prediction Error')
        plt.legend()
        plt.title('CV Graph Part 3')
        plt.show()
    print("p = 1 had the highest bias. The training error was at the highest for that graph.")
    print("p = 16 had the highest variance. The gap between the training errors were highest for that graph")
    trainError = []
    cvError = []
    for p in pList:
        maeTrain, maeVal = poly_kfoldCV(x[:50],y[:50], p, 5)
        trainError.append(maeTrain)
        cvError.append(maeVal)
    plt.plot(pList, trainError, label ='Train Error')
    plt.plot(pList, cvError, label= 'CV Error')
    plt.ylim((0,2))
    plt.xlabel('p value')
    plt.ylabel('Prediction Error')
    plt.legend()
    plt.title('CV Graph Part 3 50 samples')
    plt.show()
    print("The 2nd degree polynomial held the least CV errors")
    trainError = []
    cvError = []
    for p in pList:
        maeTrain, maeVal = poly_kfoldCV(x[:80],y[:80], p, 5)
        trainError.append(maeTrain)
        cvError.append(maeVal)
    plt.plot(pList, trainError, label ='Train Error')
    plt.plot(pList, cvError, label= 'CV Error')
    plt.ylim((0,2))
    plt.xlabel('p value')
    plt.ylabel('Prediction Error')
    plt.legend()
    plt.title('CV Graph Part 3 80 samples')
    plt.show()
    print("The 7th degree polynomial held the least CV errors")

data = pd.read_csv("data.csv")

x = data.loc[:, 'x']
y = data.loc[:, 'y']

part2()
part3()
