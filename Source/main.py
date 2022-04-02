import numpy as np
from read_file import matrix_from_file
import gaussian_jordan_elimination as jordan


def calculate_and_print_results(matrix):
    print("\nMacierz wejściowa:\n", matrix)
    try:
        result = jordan.solve_matrix(matrix)
        print("Wynik:")
        print(result[0])
        print(result[1])
    except jordan.NoneSolutionException:
        print("Układ sprzeczny")
    except jordan.InfiniteSolutionException:
        print("Układ nieoznaczony")
    print()


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

    
    user_input = ""
    while user_input != "q":
        user_input = input('''Podaj, czy chcesz obliczyć rozwiązanie układu:
        (1) Oznaczonego
        (2) Sprzecznego
        (3) Nieoznaczonego
        (4) Własnego (wczytaj z pliku)
        [q - wyjście]\n> ''')
        match(user_input):
            case '1':
                matrix = common_matrix
            case '2':
                matrix = none_results_matrix
            case '3':
                matrix = infinite_results_matrix
            case '4':
                if 'path' not in locals():
                    path = "przykladowe/i.csv"
                path = input("Podaj ścieżkę ["+path+"]: ") or path
                matrix = matrix_from_file(path)
            case 'q':
                break
            case _:
                continue
        calculate_and_print_results(matrix)
                


if __name__ == '__main__':
    main()
