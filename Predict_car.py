"""Create a Decision tree classifier in sci-kit learn using the Data given below,

features = [[2,100],[6,25],[1,300],[1,1000],[4,100],[10,100]]

Label = [1,2,1,1,2,2]

Note : 1 - Sports/Race Car and 2 - Family Car"""


import sklearn
from sklearn import tree

features = [[2,100],[6,25],[1,300],[1,1000],[4,100],[10,100]]

Label = [1,2,1,1,2,2]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,Label)
print("1 - Sports/Race Car")
print("2 - Family Car")

print(clf.predict([[4,140]]))