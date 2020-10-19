#! /usr/bin/env python3
# coding: utf-8

from src.preprocessing.clean_data import data_cleaning
from src.preprocessing.csv_to_feather import csv_to_feather


def main():
    extract_data()


def extract_data():
    csv_to_feather()
    data_cleaning()


if __name__ == "__main__":
    main()
