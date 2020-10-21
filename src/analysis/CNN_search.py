#! /usr/bin/env python3
# coding: utf-8

from keras.models import Sequential
from bayes_opt import BayesianOptimization

from src.analysis.fr_model_full import setup, train, get_sets


def main():
    searching("fr_pp")


# Hyperparameters bounds
pbounds = {
    "lr": (1e-4, 3e-4),
    "momentum": (0.45, 0.7),
    "dropout_rate": (0.03, 0.06),
    "batch_size": (200, 300),
    "dense_layers": (5, 7),
    "units": (100, 130),
}


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


def save_results(bayes_optimizer, model_name):
    print(bayes_optimizer.max)
    with open("models/best_hyperp_" + model_name, "w") as f:
        f.write(str(bayes_optimizer.max["params"]))
    with open("models/full_hyperp_" + model_name, "w") as f:
        for i, res in enumerate(bayes_optimizer.res):
            f.write("Iteration {}: \n\t{}".format(i, res))


def searching(model_name):
    bayes_optimizer = BayesianOptimization(
        f=fit_with, pbounds=pbounds, verbose=2, random_state=1
    )
    bayes_optimizer.maximize(init_points=5, n_iter=20)
    save_results(bayes_optimizer, model_name)


if __name__ == "__main__":
    main()
