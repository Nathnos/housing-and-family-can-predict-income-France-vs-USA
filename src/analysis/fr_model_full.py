import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import save_model, load_model
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


def setup(classifier, hyperparameters):
    lr, momentum, dropout_rate, dense_layers, units = hyperparameters
    classifier.add(Dense(units=units, input_dim=6, activation="relu"))
    for i in range(1, dense_layers):
        classifier.add(
            Dense(units=units * 2 ** i, input_dim=6, activation="relu")
        )
    classifier.add(Dropout(rate=dropout_rate))
    classifier.add(Dense(units=6, activation="softmax"))
    rms_opti = RMSprop(lr=lr, momentum=momentum)
    classifier.compile(
        optimizer=rms_opti,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def train(classifier, training_set, batch_size, epochs):
    X, y = training_set
    classifier.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)


def get_sets():
    X_fr_train = pd.read_feather("data/fr/train/X.feather")
    y_fr_train = pd.read_feather("data/fr/train/y.feather")
    X_fr_test = pd.read_feather("data/fr/test/X.feather")
    y_fr_test = pd.read_feather("data/fr/test/y.feather")
    return (
        (X_fr_train, get_dummy(y_fr_train)),
        (X_fr_test, get_dummy(y_fr_test)),
    )


def get_dummy(y_set):
    encoder = LabelEncoder()
    encoder.fit(y_set["PP"].values)
    encoded_Y = encoder.transform(y_set["PP"].values)
    return np_utils.to_categorical(encoded_Y, dtype=int)
