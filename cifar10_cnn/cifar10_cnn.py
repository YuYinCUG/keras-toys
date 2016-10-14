import matplotlib.pyplot as plt
from scipy.misc import imresize
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

samples_per_class = 5
num_classes = 10
data_augmentation = False
batch_size = 32
nb_epoch = 10

#Download cifar-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape: ', X_train.shape)   #('X_train shape: ', (50000, 3, 32, 32))
print(X_train.shape[0], ' train samples') #(50000, ' train samples')
print(X_test.shape[0], ' test samples')   #(10000, ' test samples')


#plot some images
for y in xrange(10):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * 10 + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        img = X_train[idx]
        img = imresize(img, (32,32,3))
        plt.imshow(img)
        plt.axis('off')
        if i == 0:
            plt.title(y)
plt.show()


#preprocessing image for training
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)
X_train /= 255
X_test /= 255


#Construct Convolution Neural Network
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model.compile(loss = 'categorical_crossentropy',
             optimizer=sgd,
             metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size = batch_size,
             nb_epoch = nb_epoch,
             validation_data = (X_test, Y_test),
             shuffle = True)
else:
    print('Using real-time data augmentation.')
    #Generate more image data for deep cnn training
    datagen = ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        samplewise_std_normalization = False,
        zca_whitening = False,
        rotation_range = 0,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        vertical_flip = False)
    datagen.fit(X_train)

    model.fit_generator(datagen.flow(X_train, Y_train,
                                    batch_size = batch_size),
                                    samples_per_epoch = X_train.shape[0],
                                    nb_epoch = nb_epoch,
                                    validation_data = (X_test, Y_test))
