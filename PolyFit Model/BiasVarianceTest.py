import GeneratingData as data
import matplotlib.pyplot as plt
import numpy as np
import random

#array of polynomials we will test
p = [1,3,5,9,15]
for y in p:
    bias = []
    for x in range(1000):
        #X and Y datasets
        dataX,dataY = data.genNoisyData()

        fit = np.polyfit(dataX, dataY, y)
        predict = np.polyval(fit,5)
        #Store all the biases
        bias.append(predict)

    #calculating the bias
    meanbias = abs(np.mean(bias))
    truebias = meanbias - data.f(5)

    #calculating the variance
    variance = np.var(bias)

    #Creating the histogram
    plt.hist(bias)
    #Creating the mean bias line
    mline = plt.axvline(meanbias, color='r')
    #Creating the f(5) line
    vline = plt.axvline(data.f(5), color='black')

    #legend of the histogram
    plt.legend([mline,vline],['mean of y$^{pred}_m$(x=5)','f(x=5)'])
    plt.xlabel('y$^{pred}_m$(x=5)')
    plt.ylabel('Counts')

    plt.show()
