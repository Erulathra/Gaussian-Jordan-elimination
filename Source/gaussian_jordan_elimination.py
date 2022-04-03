import numpy as np


def solve_matrix(augmented_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    augmented_matrix = np.array(augmented_matrix.copy())

    # Split matrix to matrix and results_vector
    results_vector: np.ndarray = augmented_matrix[:, -1]
    matrix: np.ndarray = np.delete(augmented_matrix, -1, axis=1)

    check_if_matrix_has_solution(augmented_matrix, matrix)

    # Create identity matrix
    reversed_matrix: np.ndarray = np.zeros(matrix.shape)
    np.fill_diagonal(reversed_matrix, 1)

    # sort rows
    matrix, reversed_matrix, results_vector = \
        sort_for_pivots(matrix, reversed_matrix, results_vector)

    for i in range(matrix.shape[1]):
        factor = matrix[i, i]
        # pivot cannot equal to zero
        if factor == 0:
            raise NoneSolutionException

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


def check_if_matrix_has_solution(augmented_matrix, matrix):
    # Check if matrix has solutions
    matrix_rank = np.linalg.matrix_rank(matrix)
    augmented_matrix_rank = np.linalg.matrix_rank(augmented_matrix)
    if matrix_rank != augmented_matrix_rank:
        raise NoneSolutionException
    if matrix_rank == augmented_matrix_rank and matrix_rank < matrix.shape[1]:
        raise InfiniteSolutionException


def zero_on_diagonal(matrix: np.ndarray) -> bool:
    for i in range(matrix.shape[1]):
        if (matrix[i, i]) == 0:
            return True
    return False


def sort_for_pivots(matrix: np.ndarray, reversed_matrix: np.ndarray, results_vector: np.ndarray) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    # perform column swap
    for col in range(matrix.shape[1] - 1):
        index_max_in_col = matrix[col:, col].argmax() + col
        swap_vector = [index_max_in_col, col]
        for change_matrix in (matrix, reversed_matrix, results_vector):
            change_matrix[swap_vector] = change_matrix[swap_vector[::-1]]

    return matrix, reversed_matrix, results_vector


class NoneSolutionException(Exception):
    pass


class InfiniteSolutionException(Exception):
    pass


class WrongInputException(Exception):
    pass
