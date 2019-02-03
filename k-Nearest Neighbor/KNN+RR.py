import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

#One hot encoding
def load(filepath):
    raw_data = pd.read_csv(filepath)
    #Create a data frame for column storage
    processed_columns = pd.DataFrame({})
    for col in raw_data:
        #col_datatype = raw_data[col].dtype
        #Check the column for dtype object or unique value < 20
        if raw_data[col].dtype == 'object' or raw_data[col].nunique() < 20:
            df = pd.get_dummies(raw_data[col], prefix=col)
            processed_columns = pd.concat([processed_columns, df], axis=1)
        else:
            processed_columns = pd.concat([processed_columns, raw_data[col]], axis=1)
    return processed_columns

#data = load("data_lab5.csv")

df = load("trainingset.csv")
df.drop(['rowIndex'],1, inplace=True)
df.drop(['feature1'],1, inplace=True)
df.drop(['feature2'],1, inplace=True)
df.drop(['feature6'],1, inplace=True)
df.drop(['feature8'],1, inplace=True)
df.drop(['feature10'],1, inplace=True)

td = load("testset.csv")
td.drop(['rowIndex'],1, inplace=True)
td.drop(['feature1'],1, inplace=True)
td.drop(['feature2'],1, inplace=True)
td.drop(['feature6'],1, inplace=True)
td.drop(['feature8'],1, inplace=True)
td.drop(['feature10'],1, inplace=True)

#td = np.array(td)
X = np.array(df.drop(['ClaimAmount'],1))
Y = np.array(df['ClaimAmount'])

y_classify = np.where(Y > 0, 1, 0, )


X_train, X_test, Y_train, Y_test = train_test_split(X,y_classify,test_size=0.25, shuffle=False)

clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, Y_train)
CV_mae = cross_val_score(estimator = clf, X=X_train, y=Y_train, cv = 10)
CV_mae = np.mean(CV_mae)

predictTest = clf.predict(td)
predictTrain = clf.predict(X)

rows = []
claimX_t = []
claimX = []
claimY = []

print(accuracy_score(predictTrain,y_classify))
for x in range(len(td)):
    #if clf.predict(X[x].reshape(1,-1)) == 1 and Y[x] > 0:
    if predictTest[x] == 1:
        claimX_t.append(td[x])
        rows.append(x)


#train for regression
for x in range(len(X)):
    if predictTrain[x] == 1 and Y[x] > 0:
        claimX.append(X[x])
        claimY.append(Y[x])
        

claimX = np.array(claimX).reshape(len(claimX),18)
claimY = np.array(claimY).reshape(len(claimY),)

claimX_t = np.array(claimX).reshape(len(claimX),18)
X_t_test_rr = pd.DataFrame(claimX_t)

X_train, X_test, Y_train, Y_test = train_test_split(claimX,claimY,test_size=0.25, shuffle=False)

#RIDGE REGRESSION TESTING

X_train_rr = pd.DataFrame(X_train)
X_test_rr = pd.DataFrame(X_test)
Y_train_rr = pd.DataFrame(Y_train)
Y_test_rr = pd.DataFrame(Y_test)

kFoldsX = X_train_rr
kFoldsY = Y_train_rr


maeTrain_list = []
maeValidation_list = []
size = int(len(X_train)/5)
lambdaValues = range(5,20)

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
        #maeValidation += float(np.mean(abs(CVy - CVPredict)))
        maeValidation += mean_absolute_error(CVy, CVPredict)
    print("lambda = 10^{}, CV MAE: {}".format(l, maeValidation/5))
    maeTrain_list.append(maeTrain/5)
    maeValidation_list.append(maeValidation/5)

maeValidation_list = np.array(maeValidation_list)   

print("Min index RR value = {}".format(np.argmin(maeValidation_list)))
RRMinValue = min(maeValidation_list)
RRMinIndex = np.argmin(maeValidation_list)

#Lasso Testing

maeTrain_list = []
maeValidation_list = []
size = int(len(X_train)/5)
lambdaValues2 = range(-8, 9)

#CV of 5 folds for each lambda value.
for l in lambdaValues2:
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
        
   # print("lambda = 10^{}, TRAIN MAE: {}".format((l/4.0), maeTrain/5))
    print("lambda = 10^{}, CV MAE: {}".format((l/4.0), maeValidation/5))
    maeTrain_list.append(maeTrain/5)
    maeValidation_list.append(maeValidation/5)

#lambdaManual = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
#plt.plot(lambdaManual, maeTrain_list, label ='MAE Training Data')
#plt.plot(lambdaManual, maeValidation_list, label= 'Validation Data')
#plt.xlabel('Lambda')
#plt.ylabel('MAE')
#plt.legend()
#plt.title('Lasso Graph')
#plt.show()

print("Min index Lasso value = {}".format(np.argmin(maeValidation_list)))
LMinIndex = np.argmin(maeValidation_list) 
LMinValue = min(maeValidation_list)

testRidge = Ridge(alpha=(10 ** lambdaValues[RRMinIndex]))
testRidge.fit(X_train_rr, Y_train_rr)

testPredict = testRidge.predict(X_t_test_rr)


testPredict.reshape(len(testPredict),)
emptyRows = [0]*30000

count =0 
for x in rows:
    emptyRows[x] = testPredict[count]
    count = count+1

emptyRows = np.array(emptyRows, dtype=np.float64)
submission = emptyRows
    
output = pd.DataFrame({'ClaimAmount': submission})
output.to_csv("submission.csv", index_label="rowIndex")
