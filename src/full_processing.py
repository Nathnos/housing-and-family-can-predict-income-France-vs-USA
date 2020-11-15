#! /usr/bin/env python3
# coding: utf-8

from src.preprocessing.clean_data import data_cleaning
from src.preprocessing.csv_to_feather import csv_to_feather
from src.analysis.CNN_search import search_params
import src.analysis.fr_model_full as fr_model
import subprocess


def main(search=False, train=True):
    get_data()
    extract_data()
    if search:
        search_params("fr")
        search_params("us")
    if train:
        fr_model.main(True)


def extract_data():
    csv_to_feather()
    data_cleaning()


def get_data():
    subprocess.call("get_data.sh")


if __name__ == "__main__":
    main()
