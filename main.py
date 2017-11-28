# -*- coding: utf-8 -*- #

# ---------- Import ---------- #
import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report


np.random.seed(0)


# ---------- データの準備 ---------- #
TRAIN = 10000

with open('data.pickle', 'rb') as f:
	data = pickle.load(f)

x, y = data['x'], data['y']

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size = TRAIN)

chars = '0123456789'
input_digits = 6
output_digits = 1

# ---------- モデルの設定 --------- #
n_in = len(chars)
n_hidden = 128
n_out = 2

model = Sequential()
model.add(LSTM(n_hidden, input_shape=(input_digits, n_in)))

model.add(RepeatVector(output_digits))
model.add(LSTM(n_hidden, return_sequences=True))
model.add(TimeDistributed(Dense(n_out)))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])

'''
モデル学習
'''
epochs = 2
batch_size = 500


histosy = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs,validation_data=(test_x, test_y))

model.summary()

# 検証データからランダムに問題を選んで答え合わせ
# for i in range(10):
#     index = np.random.randint(0, TEST)
#     question = test_x[np.array([index])]
#     answer = test_y[np.array([index])]
#     prediction = model.predict_classes(question, verbose=0)

#     question = question.argmax(axis=-1)
#     answer = answer.argmax(axis=-1)

#     print('-' * 10)
#     print(question[0])
#     if answer[0][0] == 1:
#             print("素数である")
#     else:
#             print("素数でない")

# print('-' * 10)

Y_pred = model.predict(test_x)
# pred_result = [bool(test[np.argmax(pred)]) for pred, test in zip(Y_pred, test_y)]
# wrong_index = np.where(np.array(pred_result) == False)
# NSLICE = 10
# 
# wrong_X = test_x[wrong_index]
# 
# fig_wrong = plt.figure(figsize=(13, 2))
# fig_wrong.suptitle("Wrong predictions")
# for i in range(NSLICE):
#     ax = fig_wrong.add_subplot(1, NSLICE, i+1)
#     ax.imshow(wrong_X[i].reshape(28,28), cmap='gray') 
#     ax.set_title(np.argmax(Y_pred[wrong_index[0][i]]))


print(classification_report(y_pred=[np.argmax(i) for i in Y_pred], y_true=[np.argmax(i) for i in test_y]))
print(Y_pred)

with open("histosy.pickle", mode='wb') as f:
        pickle.dump(histosy.history, f)

