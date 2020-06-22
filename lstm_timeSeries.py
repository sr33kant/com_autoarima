
import numpy as np
import random
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def make_list(list_length):
    num_list = []
    for count in range(list_length):
        num_list.append(random.randint(1, 101))
    return num_list
##creating a dictionary with lists of length between 6 and 13 . These lists contain integers randomly sampled between 1 and 101
data={}
for i in range(10000,20000):
    random_int = random.randint(6, 13)

    rand_list = make_list(random_int)
    data[i]=sorted(rand_list)

import pandas as pd
data_seri=pd.Series(data)
pre_data=pd.DataFrame()
#X contains sorted integers list  and y contains the last nunmber in the list
pre_data['x']=data_seri.apply(lambda x:x[:-1])
pre_data['y']=data_seri.apply(lambda x:x[-1])
# we pad these lists so all these lists are of equal length
x_padded=pad_sequences(pre_data['x'],value=0)
print(x_padded.shape)
#since this is a univariate time series we say nmber of features is 1
n_features=1
#reshaping the array into 3-d format required for lstm
X_reshaped = x_padded.reshape((x_padded.shape[0], x_padded.shape[1], n_features))
X_train,X_test=X_reshaped[0:6700,:,:],X_reshaped[6701:,:,:]
Y_train,Y_test=pre_data['y'][0:6700],pre_data[6701:]
print(X_train.shape)

model = Sequential()
model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))
#one dense layer as we are trying to predict one last integer of the sequence .
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=2)
model.predict(X_train)
model.predict(X_test)




