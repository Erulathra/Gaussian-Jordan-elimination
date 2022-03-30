import numpy as np


class NoneSolutionException(Exception):
    pass


class InfiniteSolutionException(Exception):
    pass


class WrongInputException(Exception):
    pass


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


def solve_matrix(matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    matrix = np.array(matrix.copy())

    # Split matrix to matrix and results_vector
    results_vector: np.ndarray = matrix[:, -1]
    matrix: np.ndarray = np.delete(matrix, -1, axis=1)

    # Create identity matrix
    reversed_matrix: np.ndarray = np.zeros(matrix.shape)
    np.fill_diagonal(reversed_matrix, 1)

    if results_vector.shape[0] < matrix.shape[0]:
        raise InfiniteSolutionException

    if np.linalg.det(matrix) == 0:
        if check_is_result_has_multiples(matrix, results_vector):
            raise InfiniteSolutionException
        else:
            raise NoneSolutionException

    for i in range(matrix.shape[1]):
        factor = matrix[i, i]

        # Transform row to have one on diagonal
        matrix[i] /= factor
        results_vector[i] /= factor
        reversed_matrix[i] /= factor
        # Subtract this row from rest of the rows
        for j in range(i + 1, matrix.shape[1]):
            factor = matrix[j, i]
            matrix[j] -= matrix[i] * factor
            results_vector[j] -= results_vector[i] * factor
            reversed_matrix[j] -= reversed_matrix[i] * factor

    # Subtract rows from end to make matrix diagonal
    for i in range(matrix.shape[1])[::-1]:
        for j in range(i):
            factor = matrix[j, i]
            matrix[j] -= matrix[i] * factor
            results_vector[j] -= results_vector[i] * factor
            reversed_matrix[j] -= reversed_matrix[i] * factor

    return results_vector, reversed_matrix
