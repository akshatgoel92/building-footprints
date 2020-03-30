# logistic regression with L1 and L2 regularization
from keras.regularizers import l1l2
from keras.models import Sequential
from keras.layers import Dense


def load_data():
    pass


def model_compile():
    # 2-class logistic regression in Keras
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=x.shape[1]))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')


def model_fit():
    model.fit(x, y, nb_epoch=10, validation_data=(x_val, y_val))