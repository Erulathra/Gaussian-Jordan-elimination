import numpy as np
import csv

def matrix_from_file(path) -> np.array:
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        result = []
        for line in reader:
            result.append([int(char) for char in line])
    return np.array(result)


#DELETE THIS
arr = matrix_from_file("file.csv")
print(arr)