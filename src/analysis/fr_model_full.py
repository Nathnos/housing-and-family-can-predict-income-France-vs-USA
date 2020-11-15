import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import save_model, load_model
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import json


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


def train(classifier, training_set, batch_size, epochs, verbose=0):
    X, y = training_set
    classifier.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)


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


def main(new_model):
    if new_model:
        model = train_new_model()
        save_model(model, "models/fr_full")
    model = load_model("models/fr_full")
    print(evaluate(model))


def evaluate(model):
    _, test_set = get_sets()
    X, y = test_set
    score = model.evaluate(x=X, y=y, steps=10, verbose=0)
    return score


def train_new_model():
    with open("models/best_hyperp_fr_pp") as hyper_params_file:
        json_object = json.loads(hyper_params_file.read().replace("\'", "\""))
    hyperparameters = (
        float(json_object.get("lr")),
        float(json_object.get("momentum")),
        float(json_object.get("dropout_rate")),
        int(json_object.get("dense_layers")),
        int(json_object.get("units")),
    )
    batch_size = int(json_object.get("batch_size"))
    classifier = Sequential()
    training_set, _ = get_sets()
    setup(classifier, hyperparameters)
    epochs = 200
    train(classifier, training_set, batch_size, epochs, verbose=1)
    return classifier
