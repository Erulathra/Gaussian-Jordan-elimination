import numpy as np
import gaussian_jordan_elimination as jordan


def main():
    common_matrix = np.array([[3, 3, 1, 12],
                              [2, 5, 7, 33],
                              [1, 2, 1, 8]], float)

    infinite_results_matrix = np.array([[3, 3, 1, 1],
                                        [2, 5, 7, 20],
                                        [-4, -10, -14, -40]], float)

    none_results_matrix = np.array([[3, 3, 1, 20],
                                    [2, 5, 7, 20],
                                    [-4, -10, -14, -20]], float)

    result = jordan.solve_matrix(common_matrix)
    print(result)

    result = jordan.solve_matrix(infinite_results_matrix)
    print(result)

    result = jordan.solve_matrix(none_results_matrix)
    print(result)

    A = np.array([2, 4, 6])
    B = np.array([1, 2, 3])


if __name__ == '__main__':
    main()
