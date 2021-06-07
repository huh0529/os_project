import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import os
from datetime import datetime, date
import tensorflow as tf


df =  pd.read_csv('predict.csv')

MAX = df['Infected'].max()
MAX1 = df['Test_number'].max()
MAX2 = df['spread'].max()
MAX3 = df['policy'].max()


scaler = MinMaxScaler()
scale_cols = ['Test_number', 'spread', 'policy', 'Infected']
df_scaled = scaler.fit_transform(df[scale_cols])
xy = df_scaled
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

def make_dataset2(data, window_size=20):
    label_list = []
    for i in range(len(data) - window_size):
        label_list.append(data[i:i+window_size])
    return np.array(label_list)

SIZE_RATE = 0.8
TEST_SIZE = int(len(df_scaled)*SIZE_RATE)
WINDOW_SIZE = 20

train = df_scaled[:TEST_SIZE]
test = df_scaled[TEST_SIZE:]

feature_cols = ['Test_number', 'spread', 'policy']
label_cols = ['Infected']

train_feature = train[feature_cols]
train_label = train[label_cols]

train_feature, train_label = make_dataset(train_feature, train_label, 20)

x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
x_train.shape, x_valid.shape

test_feature = test[feature_cols]
test_label = test[label_cols]

test_feature.shape, test_label.shape

test_feature, test_label = make_dataset(test_feature, test_label, 20)
test_feature.shape, test_label.shape

model = Sequential()
model.add(LSTM(16,
               input_shape=(train_feature.shape[1], train_feature.shape[2]),
               activation='relu',
               return_sequences=False)
          )

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model_path = 'model'
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train,
                                    epochs=200,
                                    batch_size=16,
                                    validation_data=(x_valid, y_valid),
                                    callbacks=[early_stop, checkpoint])


model.load_weights(filename)
##############################################################################
feature_cols2 = ['Infected']
label_cols2 = ['Test_number']

train_feature2 = train[feature_cols2]
train_label2 = train[label_cols2]

train_feature2, train_label2 = make_dataset(train_feature2, train_label2, 20)

x_train2, x_valid2, y_train2, y_valid2 = train_test_split(train_feature2, train_label2, test_size=0.2)
x_train2.shape, x_valid2.shape

test_feature2 = test[feature_cols2]
test_label2 = test[label_cols2]

test_feature2.shape, test_label2.shape

test_feature2, test_label2 = make_dataset(test_feature2, test_label2, 20)
test_feature2.shape, test_label2.shape

model2 = Sequential()
model2.add(LSTM(16,
               input_shape=(train_feature2.shape[1], train_feature2.shape[2]),
               activation='relu',
               return_sequences=False)
          )

model2.add(Dense(1))

model2.compile(loss='mean_squared_error', optimizer='adam')
early_stop2 = EarlyStopping(monitor='val_loss', patience=5)

model_path2 = 'model2'
filename2 = os.path.join(model_path2, 'tmp_checkpoint.h5')
checkpoint2 = ModelCheckpoint(filename2, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history2 = model2.fit(x_train2, y_train2,
                                    epochs=200,
                                    batch_size=16,
                                    validation_data=(x_valid2, y_valid2),
                                    callbacks=[early_stop2, checkpoint2])


model2.load_weights(filename2)
########################################################################
feature_cols3 = ['Infected']
label_cols3 = ['spread']

train_feature3 = train[feature_cols3]
train_label3 = train[label_cols3]

train_feature3, train_label3 = make_dataset(train_feature3, train_label3, 20)

x_train3, x_valid3, y_train3, y_valid3 = train_test_split(train_feature3, train_label3, test_size=0.2)
x_train3.shape, x_valid3.shape

test_feature3 = test[feature_cols3]
test_label3 = test[label_cols3]

test_feature3.shape, test_label3.shape

test_feature3, test_label3 = make_dataset(test_feature3, test_label3, 20)
test_feature3.shape, test_label3.shape

model3 = Sequential()
model3.add(LSTM(16,
               input_shape=(train_feature3.shape[1], train_feature3.shape[2]),
               activation='relu',
               return_sequences=False)
          )

model3.add(Dense(1))

model3.compile(loss='mean_squared_error', optimizer='adam')
early_stop3 = EarlyStopping(monitor='val_loss', patience=5)

model_path3 = 'model2'
filename3 = os.path.join(model_path3, 'tmp_checkpoint.h5')
checkpoint3 = ModelCheckpoint(filename3, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history3 = model3.fit(x_train3, y_train3,
                                    epochs=200,
                                    batch_size=16,
                                    validation_data=(x_valid3, y_valid3),
                                    callbacks=[early_stop3, checkpoint3])


model2.load_weights(filename2)

pred = model.predict(test_feature)
pred.shape


test_label *= MAX
pred*=MAX


plt.figure(figsize=(12, 9))
plt.plot(test_label, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.xticks(rotation=45)
plt.legend()
plt.show()

########################################################################

pol = 2
t_p = np.array([[pol]])
t_p=t_p.astype(np.float)

for i in range(0,7):

    a = test_feature[[-1]]
    b = model.predict(a)

    t1_i = np.array(test[label_cols])
    t1_i=np.concatenate((t1_i, b))
    t1_i.shape
    t1_i = make_dataset2(t1_i, 20)
    t1_i.shape
    t1_i=t1_i[[-1]]

    t1_t = model2.predict(t1_i)
    t1_s = model3.predict(t1_i)

    arr = np.concatenate((t1_t, t1_s, t_p, b), axis=1)

    test = scaler.fit_transform(test[scale_cols])
    test = np.concatenate((test, arr))
    test = pd.DataFrame(test)
    test.columns = scale_cols

    test_feature = test[feature_cols]
    test_label = test[label_cols]

    test_feature.shape, test_label.shape

    test_feature, test_label = make_dataset(test_feature, test_label, 20)
    test_feature.shape, test_label.shape

pred = model.predict(test_feature)
pred.shape

list2 = pd.read_csv('inf.csv',index_col=0)

test_label *= MAX
pred*=MAX
list2['Infected']=pred

list=pd.read_csv('predict.csv', index_col=0)
del list['spread']
del list['policy']
del list['Test_number']

#list.loc['2021-05-29':].to_csv('list.csv')
#list2.loc['2021-05-29':].to_csv('list2.csv')


plt.figure(figsize=(12, 9))
plt.plot(list.loc['2021-05-29':], label = 'actual')
plt.plot(list2.loc['2021-05-29':], label = 'prediction')
plt.xticks(rotation=45)
plt.ylim(0, 1000)
plt.legend()
plt.show()
