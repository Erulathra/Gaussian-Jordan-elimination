import numpy as np
import csv


def matrix_from_file(path) -> np.array:
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        result = []
        for line in reader:
            result.append([float(char) for char in line])
    return np.array(result, float)


def matrix_from_user_input() -> np.array:
    string_matrix = input('''Wprowadź macierz, oddzielając pola spacjami, a linie średnikami, np:\n\
        3 4 5 1; 2 5 8 4; 9 2 3 5\n> ''')
    string_matrix = [line.split() for line in string_matrix.split(";")]
    return np.array(string_matrix, float)
