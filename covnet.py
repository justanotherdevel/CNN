from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import pickle

X_train, y_train = pickle.load(open('/home/shashwat/Programs/Python/CNN/data.p', 'rb'))

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, batch_size=100)

print(history_object.history.keys())

model.save('model.h5')
