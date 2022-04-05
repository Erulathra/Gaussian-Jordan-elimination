import numpy as np
from read_input import matrix_from_file, matrix_from_user_input
import gaussian_jordan_elimination as jordan


def main():
    user_input = ""
    path = "../przykladowe/e.csv"
    while user_input != "q":
        try:
            user_input = input('''Podaj, czy chcesz obliczyć rozwiązanie układu:
            (1) Oznaczonego
            (2) Sprzecznego
            (3) Nieoznaczonego
            (4) Predefinowanego
            (5) Własnego (wpisz ręcznie)
            (6) Własnego (wczytaj z pliku)
            [q - wyjście]\n> ''')
            match user_input:
                case '1':
                    matrix = Examples.common_matrix
                case '2':
                    matrix = Examples.none_results_matrix
                case '3':
                    matrix = Examples.infinite_results_matrix
                case '4':
                    letter = input(f"Podaj przykład (a-j): ")
                    matrix = matrix_from_file(f"../przykladowe/{letter}.csv")
                case '5':
                    matrix = matrix_from_user_input()
                case '6':
                    path = input(f"Podaj ścieżkę: ")
                    matrix = matrix_from_file(path)
                case 'q':
                    break
                case _:
                    continue
            calculate_and_print_results(matrix)
        except ValueError:
            print("Podczas podawania parsowania macierzy wystąpił błąd. Sprawdź poprawność wprowadzanych danych.")


class Examples:
    common_matrix = np.array([[3, 3, 1, 12],
                              [2, 5, 7, 33],
                              [1, 2, 1, 8]], float)

    infinite_results_matrix = np.array([[3, 3, 1, 1],
                                        [2, 5, 7, 20],
                                        [-4, -10, -14, -40]], float)

    none_results_matrix = np.array([[3, 3, 1, 20],
                                    [2, 5, 7, 20],
                                    [-4, -10, -14, -20]], float)


def calculate_and_print_results(matrix):
    print("\nMacierz wejściowa:\n", matrix)
    try:
        result = jordan.solve_matrix(matrix)
        print("Wynik:")
        print(result[0])
        print("Macierz odwrócona:")
        print(result[1])
    except jordan.NoneSolutionException:
        print("Układ sprzeczny")
    except jordan.InfiniteSolutionException:
        print("Układ nieoznaczony")
    except jordan.DeterminantIsZeroException:
        print("Macierz ma wyznacznik równy zero, nie można obliczyć macierzy odwrotnej")
    except jordan.WrongInputException:
        print("Macierz wejsciowa nie jest kwadratowa")
    print()
    input("Naciśnij Enter aby kontynuować")


if __name__ == '__main__':
    main()
