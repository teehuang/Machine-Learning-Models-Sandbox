import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error


data = pd.read_csv("data.csv")


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Setup marker generator and color map.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot all samples.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

def step9():
    X = data.drop(['digit'], axis=1, inplace=False)
    Y = data.loc[:,['digit']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False)

    pca = PCA(n_components=2)
    transformedTrain = pca.fit_transform(X_train)
    transformedTest = pca.transform(X_test)

    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    clf.fit(transformedTrain,y_train)
    transformedPredict = clf.predict(transformedTest)
    y_test = np.ravel(y_test)

    #error rate
    count = 0
    for e in range(len(transformedPredict)):
        if y_test[e] == transformedPredict[e]:
            count = count + 1

    error = 1-count/len(transformedPredict)

    print("Error rate: {}".format(error))
    plt.xlabel('Transformed Features')
    plt.ylabel('Y Test')
    plt.title('Transformed KNN 2 Feature Graph')
    plot_decision_regions(transformedTest, transformedPredict, clf)
    plt.legend()
    plt.show()


def step10():
    X = data.drop(['digit'], axis=1, inplace=False)
    Y = data.loc[:,['digit']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False)

    pca = PCA(n_components=None)
    pca.fit(X_train)
    scree = pca.explained_variance_ratio_
    plt.plot(range(len(scree)), scree)
    plt.xlabel('Number of Features')
    plt.ylabel('Scree')
    plt.title('Scree Graph')
    plt.legend()
    plt.show()
    print("Elbow is around 10 features")

def step12():
    X = data.drop(['digit'], axis=1, inplace=False)
    Y = data.loc[:,['digit']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False)

    for x in range(15):
        pca = PCA(n_components=x+1)
        transformedTrain = pca.fit_transform(X_train)
        transformedTest = pca.transform(X_test)

        clf = neighbors.KNeighborsClassifier(n_neighbors=5)
        clf.fit(transformedTrain,y_train)
        transformedPredict = clf.predict(transformedTest)
        y_test = np.ravel(y_test)

        #error rate
        count = 0
        for e in range(len(transformedPredict)):
            if y_test[e] == transformedPredict[e]:
                count = count + 1

        error = 1-count/len(transformedPredict)

        print("PC = {}. Error rate: {}".format(x+1, error))
    print("We can reduce to 10 features")


step9()
step10()
step12()
