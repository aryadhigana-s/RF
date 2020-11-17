import numpy as np
import random as rd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

data = np.genfromtxt('angka.csv', delimiter=',')
dataTA = pd.read_csv('kelas.csv')
# print(dataTrain)

#np.random.shuffle(data)

np.random.seed(1)
rd.seed(1)
dataTrain = np.nan_to_num(data[0:70,:-1])
label_dataTrain = np.nan_to_num(data[0:70,-1])
dataTest = np.nan_to_num(data[70:,:-1])
label_dataTest = np.nan_to_num(data[70:,-1])

# data = np.genfromtxt('angka2.csv', delimiter=',')

np.random.seed(1)

features = np.nan_to_num(data[0:,:-1]) #input
labels = np.nan_to_num(data[0:,-1]) #output

skf = StratifiedKFold(n_splits=5)

clf = RandomForestClassifier(n_estimators=100, max_depth=10)

clf.fit(dataTrain, label_dataTrain)
ans = clf.predict(dataTest)
ans2 = clf.predict(dataTrain)
print ("Hasil Prediksi Kelas: ",ans2, ans)
print("Akurasi: ",accuracy_score(ans,label_dataTest))
print("nilai presisi: ",precision_score(ans, label_dataTest, average='micro'))
print("nilai recall: ",recall_score(ans, label_dataTest, average='micro'))

print("Pengujian dengan 5-fold cross-validation")
fold = 1
for train_index, test_index in skf.split(features, labels):
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    print("fold {}".format(fold), end="\t")
    print("banyak data latih {}".format(len(y_train)), end="\t")
    print("banyak data uji {}".format(len(y_test)), end="\t")
    print("")
    clf.fit (x_train, y_train)
    ans = clf.predict(x_test)
    clf.fit(x_train, y_train)
    print("akurasi: {}".format(clf.score(x_test,y_test)), end="\t")
    print("")
    fold += 1

# ans = clf.predict(dataTest)
# ans2 = clf.predict(dataTrain)
# print(ans)
# print("Precision: ",precision_score(ans, label_dataTest, average = 'micro'))
# print("Recall: ",recall_score(ans,label_dataTest, average = 'micro'))