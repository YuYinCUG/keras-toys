import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
def load_data():
    '''loading train data and test data for training cnn.
    '''

    train_csv = pd.read_csv('train.csv')
    test_csv = pd.read_csv('test.csv')

    train = train_csv.iloc[:,1:].values
    train_label = train_csv[[0]].values.ravel()
    test = test_csv.iloc[:,:].values

    train = train.reshape(train.shape[0],1,28,28).astype('float32')
    test = test.reshape(test.shape[0],1,28,28).astype('float32')
    train = train / 255
    test = test / 255
    print('Train Data\' Shape: {}'.format(train.shape))
    print('Train Label Data\' Shape:{}'.format(train_label.shape))
    print('Test Data\' Shape: {}'.format(test.shape))
    return train, train_label, test


def run_model(train, train_label, test):
    '''training cnn model for predicting the label of test data
    '''

    # for calculating cross entropy
    train_label = np_utils.to_categorical(train_label)

    model = Sequential()
    model.add(Convolution2D(16,3,3, border_mode = 'valid',input_shape = (1,28,28),activation = 'relu'))
    model.add(Convolution2D(16,3,3, border_mode = 'valid',input_shape = (1,28,28),activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(32,3,3, border_mode = 'same',activation = 'relu'))
    model.add(Convolution2D(32,3,3, border_mode = 'same',activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64,3,3, border_mode = 'same', activation = 'relu'))
    model.add(Convolution2D(64,3,3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    sgd = SGD(lr = 0.01, momentum = 0.9, decay = 1e-6, nesterov = True)
    model.compile(loss='categorical_crossentropy',optimizer = sgd, metrics = ['accuracy'])
    model.fit(train, train_label, nb_epoch = 100, batch_size = 200, verbose = 2,validation_split = 0.2)

    pred = model.predict(test)
    pred = pred.argmax(axis=1)

    return pred

def save_submission(pred, filename):
    pred_pd = pd.DataFrame()
    pred_pd['ImageId'] = np.arange(1, len(pred)+1)
    pred_pd['Label'] = pred
    pred_pd.to_csv(filename+'.csv', index=False)

    print('Save submission in {}'.format(filename+'.csv'))

if __name__ == '__main__':
    train, train_label, test = load_data()
    pred = run_model(train, train_label, test)

    import time
    filename = time.asctime()
    save_submission(pred, filename)
