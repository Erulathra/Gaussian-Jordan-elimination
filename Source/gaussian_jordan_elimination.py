import numpy as np


# czy trzeba macierz odwrotną w Jordanie;
# czy trzeba umieć liczyć macierze nie kwadratowe;
class EquationType:
    unique = "Unique"
    infinite = "Infinite"
    none = "None"


def check_is_result_has_multiples(matrix: np.ndarray, result: np.ndarray) -> bool:
    glued_matrix = matrix
    glued_matrix = np.append(glued_matrix, np.matrix(result).T, axis=1)
    glued_matrix = np.array(glued_matrix)

    for i in range(result.size):
        for j in range(i + 1, result.size):
            correlation = glued_matrix[i] / glued_matrix[j]
            if np.all(correlation == correlation[0]):
                return True

    return False


def solve_matrix(matrix: np.ndarray) -> (EquationType, np.ndarray):
    matrix = np.array(matrix.copy())

    result_vector: np.ndarray = matrix[:, -1]
    matrix: np.ndarray = np.delete(matrix, -1, axis=1)

    if result_vector.shape[0] < matrix.shape[0]:
        return EquationType.infinite

    if np.linalg.det(matrix) == 0:
        if check_is_result_has_multiples(matrix, result_vector):
            return EquationType.infinite, result_vector
        else:
            return EquationType.none, result_vector

    for i in range(matrix.shape[1]):
        factor = matrix[i, i]

        # Transform row to have one on diagonal
        matrix[i] /= factor
        result_vector[i] /= factor
        # Subtract this row from rest of the rows
        for j in range(i + 1, matrix.shape[1]):
            factor = matrix[j, i]
            matrix[j] -= matrix[i] * factor
            result_vector[j] -= result_vector[i] * factor

    # Subtract rows from end to make matrix diagonal
    for i in range(matrix.shape[1])[::-1]:
        for j in range(i):
            factor = matrix[j, i]
            matrix[j] -= matrix[i] * factor
            result_vector[j] -= result_vector[i] * factor

    return EquationType.unique, result_vector
