import pandas
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from pre_process import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

df = pandas.read_csv('data_moods.csv')

X = df[df.columns[6:-3]]
Y = df['mood']

X = MinMaxScaler().fit_transform(X)
X2 = np.array(df[df.columns[6:-3]])
encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)

train_X, test_X, train_Y, test_Y = train_test_split(X, encoded_y, test_size=.2, random_state=15)

target = pandas.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)


def modelo():
    model = Sequential()
    model.add(Dense(8,input_dim=10,activation='relu'))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# estimator = KerasClassifier(build_fn = modelo, epochs = 400, batch_size = 200)

# k_fold = KFold(n_splits = 10, shuffle = True)
# res = cross_val_score(estimator, X, encoded_y, cv = k_fold)
# print("Accuracy of model: %.2f%%, standard deviation: %.2f%%" % (res.mean()*100, res.std()*100))


# estimator.fit(X_train,Y_train)
# y_preds = estimator.predict(X_test)

# cm = confusion_matrix(Y_test,y_preds)
# ax = plt.subplot()
# seaborn.heatmap(cm,annot=True,ax=ax)

# labels = target['mood']
# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# ax.set_title('Confusion Matrix')
# ax.xaxis.set_ticklabels(labels)
# ax.yaxis.set_ticklabels(labels)
# plt.show()