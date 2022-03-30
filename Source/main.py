import numpy as np
import gaussian_jordan_elimination as jordan


def calculate_and_print_results(matrix):
    try:
        result = jordan.solve_matrix(matrix)
        print(result[0])
        print(result[1])
    except jordan.NoneSolutionException:
        print("Układ sprzeczny")
    except jordan.InfiniteSolutionException:
        print("Układ nieoznaczony")


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

    calculate_and_print_results(common_matrix)
    calculate_and_print_results(infinite_results_matrix)
    calculate_and_print_results(none_results_matrix)


if __name__ == '__main__':
    main()
