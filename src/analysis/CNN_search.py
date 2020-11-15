#! /usr/bin/env python3
# coding: utf-8

from bayes_opt import BayesianOptimization

from src.analysis.CNN_search_helper import get_fit_with


def search_params(state):
    searching(state, get_fit_with(state))


# Hyperparameters bounds
pbounds = {
    "lr": (1e-4, 1e-1),
    "momentum": (0.3, 0.9),
    "dropout_rate": (0.00, 0.5),
    "batch_size": (200, 2000),
    "dense_layers": (1, 3),
    "units": (12, 64),
}


def save_results(bayes_optimizer, model_name):
    print(bayes_optimizer.max)
    with open("models/best_hyperp_" + model_name, "w") as f:
        f.write(str(bayes_optimizer.max["params"]))
    with open("models/full_hyperp_" + model_name, "w") as f:
        for i, res in enumerate(bayes_optimizer.res):
            f.write("Iteration {}: \n\t{}".format(i, res))


def searching(model_name, fit_with):
    bayes_optimizer = BayesianOptimization(
        f=fit_with, pbounds=pbounds, verbose=2, random_state=1
    )
    bayes_optimizer.maximize(init_points=10, n_iter=30)
    save_results(bayes_optimizer, model_name)


if __name__ == "__main__":
    search_params("fr_sl")
