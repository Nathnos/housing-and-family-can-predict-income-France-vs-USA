#! /usr/bin/env python3
# coding: utf-8

from keras.models import Sequential

import src.analysis.fr_model_full as fr


def get_fit_with(state):
    if state == "fr":
        get_sets = fr.get_sets
        setup = fr.setup
        train = fr.train

    # Let's do a closure !
    def fit_with(lr, momentum, dropout_rate, batch_size, dense_layers, units):
        hyperparameters = (
            lr,
            momentum,
            dropout_rate,
            int(dense_layers),
            int(units),
        )
        batch_size = int(batch_size)
        classifier = Sequential()
        training_set, test_set = get_sets()
        setup(classifier, hyperparameters)
        epochs = 50
        train(classifier, training_set, batch_size, epochs)
        X, y = test_set
        score = classifier.evaluate(x=X, y=y, steps=10, verbose=0)
        return score[1]  # Accuracy

    return fit_with
