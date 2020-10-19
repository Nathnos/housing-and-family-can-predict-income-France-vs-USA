import pandas as pd
from sklearn import preprocessing
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.models import save_model, load_model

X_fr_train = pd.read_feather("data/fr/train/X.feather")
y_fr_train = pd.read_feather("data/fr/train/y.feather")
X_us_train = pd.read_feather("data/us/train/X.feather")
y_us_train = pd.read_feather("data/us/train/y.feather")
scaler = preprocessing.StandardScaler().fit(X_us_train)
X_fr_train = scaler.transform(X_fr_train)
X_us_train = scaler.transform(X_us_train)

def train():
    model = Sequential()
    model.add(Dense(units=128, input_dim=5, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_us_train, y_us_train, batch_size=20, epochs=10)
    save_model(model, "models/fr_full")


def test():
    X_fr_test = scaler.transform(pd.read_feather("data/fr/test/X.feather"))
    y_fr_test = pd.read_feather("data/fr/test/y.feather")
    X_us_test = scaler.transform(pd.read_feather("data/us/test/X.feather"))
    y_us_test = pd.read_feather("data/us/test/y.feather")
    model = load_model("models/fr_full")
    # loss, acc = model.evaluate(X_fr_test, y_fr_test, verbose=0)
    # print(loss, acc)
    loss, acc = model.evaluate(X_us_test, y_us_test, verbose=0)
    print(loss, acc)
    i = 147
    print(model.predict(X_us_test)[i:i+10], y_us_test.values[i:i+10])
    # print(np.concatenate(model.predict(X_us_test), y_us_test.values))


train()
test()
