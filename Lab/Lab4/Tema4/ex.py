# import numpy as np
# import urllib.request
# import time

# # Funcția care încarcă matricea rară din fișierul dat prin link
# def load_sparse_matrix_from_link(link, epsilon=1e-6):
#     try:
#         content_a = urllib.request.urlopen(link).read().decode('utf-8').split('\n')
#         n = int(content_a[0])
#         valori = []
#         ind_col = []
#         inceput_linii = [0]

#         for line in content_a[1:]:
#             val = line.split(',')
#             if len(val) != 3:
#                 continue
#             x = float(val[0].strip())
#             if abs(x) < epsilon:
#                 x = 0  # Valorile mai mici decât epsilon sunt considerate 0
#             i = int(val[1].strip())
#             j = int(val[2].strip())
#             valori.append(x)
#             ind_col.append(j)
#             if i > len(inceput_linii) - 1:
#                 inceput_linii.append(len(valori))
#         return np.array(valori), np.array(ind_col), np.array(inceput_linii)
#     except Exception as e:
#         print("Eroare la încărcarea matricei rare:", e)
#         return None, None, None

# # Funcția care încarcă datele pentru sistemul liniar din fișierele a și b pentru un anumit index
# def load_system(file_index):   #asta o facem cu compresed lines
#     try:
#         base_url = "https://profs.info.uaic.ro/~ancai/CN/lab/4/sislinrar/"
#         file_a = f"{base_url}a_{file_index}.txt"
#         file_b = f"{base_url}b_{file_index}.txt"
        
#         valori, ind_col, inceput_linii = load_sparse_matrix_from_link(file_a)
#         content_b = urllib.request.urlopen(file_b).read().decode('utf-8').split('\n')
#         b_vector = [float(val) for val in content_b[1:] if len(val.strip()) > 0]
        
#         return valori, ind_col, inceput_linii, b_vector
#     except Exception as e:
#         print(f"Eroare la încărcarea sistemului liniar pentru indexul {file_index}:", e)
#         return None, None, None, None
    
# def load_system_bcrs(file_index, block_size):
#     try:
#         valori, ind_col, inceput_linii, b = load_system(file_index)
#         if valori is None or ind_col is None or inceput_linii is None or b is None:
#             return None, None, None, None

#         n = len(inceput_linii) - 1
#         block_count = n // block_size
#         block_dimension = block_size ** 2
        
#         val_bcrs = np.zeros((block_count, block_dimension))
#         col_ind_bcrs = np.zeros(block_count * block_dimension, dtype=int)
#         row_blk_bcrs = np.zeros(block_count + 1, dtype=int)

#         for block_index in range(block_count):
#             row_blk_bcrs[block_index] = block_index * block_dimension
#             for i in range(block_size):
#                 for j in range(block_size):
#                     val_bcrs[block_index, i * block_size + j] = valori[inceput_linii[block_index] + i * n + j]
#                     col_ind_bcrs[block_index * block_dimension + i * block_size + j] = ind_col[inceput_linii[block_index] + i * n + j]
#         row_blk_bcrs[block_count] = block_count * block_dimension

#         return val_bcrs, col_ind_bcrs, row_blk_bcrs, b
#     except Exception as e:
#         print(f"Eroare la încărcarea sistemului liniar în format BCRS pentru indexul {file_index}:", e)
#         return None, None, None, None    
# # vezi    
# def gauss_seidel_sparse(A, b, inceput_linii=None, epsilon=1e-6, max_iter=1000):
#     n = len(b)
#     x = np.zeros(n)
#     x_new = np.zeros(n)
#     iter_count = 0
    
#     start_time = time.time()
    
#     while iter_count < max_iter:
#         for i in range(n):
#             if inceput_linii is not None:  # Verificați dacă există indici de început al liniilor
#                 start_idx = inceput_linii[i]
#                 end_idx = inceput_linii[i + 1] if i + 1 < len(inceput_linii) else len(A)
#                 sum1 = np.dot(A[start_idx:end_idx], x_new)
#                 sum2 = np.dot(A[start_idx:end_idx, i + 1:], x[i + 1:])
#                 x_new[i] = (b[i] - sum1 - sum2) / A[start_idx, i]
#             else:
#                 sum1 = np.dot(A[i], x_new)
#                 sum2 = np.dot(A[i + 1:], x[i + 1:])
#                 x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
            
#         if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
#             break
        
#         x = np.copy(x_new)
#         iter_count += 1
    
#     end_time = time.time()
#     execution_time = end_time - start_time
    
#     return x, iter_count, execution_time

# # Iterăm peste toate indexurile de la 1 la 5 și încărcăm sistemele liniare
# for i in range(1, 6):
# #test meth 1
#     # print(f"\nÎncărcăm sistemul liniar cu indexul {i}:")
#     # valori, ind_col, inceput_linii, b = load_system(i)
#     # if valori is not None and ind_col is not None and inceput_linii is not None and b is not None:
#     #     print("Valori nenule ale matricei rare:")
#     #     print(valori)
#     #     print("Indicii de coloană ai matricei rare:")
#     #     print(ind_col)
#     #     print("Vectorul de început al liniilor:")
#     #     print(inceput_linii)
#     #     print("Vectorul termenilor liberi b:")
#     #     print(b)
#     # else:
#     #     print("Nu s-a putut încărca sistemul liniar.")
# #test meth 2
#     # print(f"\nÎncărcăm sistemul liniar cu indexul {i} în formatul BCRS:")
#     # block_size = 3  # Dimensiunea blocului pentru BCRS (poate fi modificată)
#     # val_bcrs, col_ind_bcrs, row_blk_bcrs, b_bcrs = load_system_bcrs(i, block_size)
#     # if val_bcrs is not None and col_ind_bcrs is not None and row_blk_bcrs is not None and b_bcrs is not None:
#     #     print("Valori blocuri matrice rară (BCRS):")
#     #     print(val_bcrs)
#     #     print("Indici coloană blocuri matrice rară (BCRS):")
#     #     print(col_ind_bcrs)
#     #     print("Pointeri început blocuri pe linii (BCRS):")
#     #     print(row_blk_bcrs)
#     #     print("Vectorul termenilor liberi b:")
#     #     print(b_bcrs)
#     # else:
#     #     print("Nu s-a putut încărca sistemul liniar în formatul BCRS.")
#     #     exit()

#     print(f"\nAproximăm soluția sistemului liniar cu indexul {i} folosind metoda Gauss-Seidel:")
#     for method in ['CSR', 'BCRS']:
#         print(f"\nMetoda de memorare rară: {method}")
#         if method == 'CSR':
#             valori, ind_col, inceput_linii, b = load_system(i)
#             if valori is not None and ind_col is not None and inceput_linii is not None and b is not None:
#                 solutie_aproximativa, numar_iteratii, timp_executie = gauss_seidel_sparse(valori, b, inceput_linii)
#                 print("Soluție aproximativă:")
#                 print(solutie_aproximativa)
#                 print("Numărul de iterații necesare:")
#                 print(numar_iteratii)
#                 print("Timpul de execuție (secunde):")
#                 print(timp_executie)
                
#                 # Calcularea soluției exacte cu Gauss-Seidel
#                 solutie_exacta, _, _ = gauss_seidel_sparse(valori, b, inceput_linii, epsilon=1e-15)
                
#                 # Calcularea normei diferenței dintre soluția aproximativă și cea exactă
#                 norma_diferenta = np.linalg.norm(solutie_aproximativa - solutie_exacta, ord=np.inf)
#                 print("Norma diferenței între soluția aproximativă și cea exactă:")
#                 print(norma_diferenta)
#             else:
#                 print("Nu s-a putut încărca sistemul liniar.")
#         else:
#             block_size = 3
#             valori, ind_col, inceput_linii, b = load_system_bcrs(i,block_size)
#             if valori is not None and ind_col is not None and inceput_linii is not None and b is not None:
#                 solutie_aproximativa, numar_iteratii, timp_executie = gauss_seidel_sparse(valori, b, inceput_linii)
#                 print("Soluție aproximativă:")
#                 print(solutie_aproximativa)
#                 print("Numărul de iterații necesare:")
#                 print(numar_iteratii)
#                 print("Timpul de execuție (secunde):")
#                 print(timp_executie)
                
#                 # Calcularea soluției exacte cu Gauss-Seidel
#                 solutie_exacta, _, _ = gauss_seidel_sparse(valori, b, inceput_linii, epsilon=1e-15)
                
#                 # Calcularea normei diferenței dintre soluția aproximativă și cea exactă
#                 norma_diferenta = np.linalg.norm(solutie_aproximativa - solutie_exacta, ord=np.inf)
#                 print("Norma diferenței între soluția aproximativă și cea exactă:")
#                 print(norma_diferenta)
#             else:
#                 print("Nu s-a putut încărca sistemul liniar.")
import numpy as np
import math


def verify_divide(number):
    if abs(number) <= 10 ** (-6):
        raise ValueError("CANNOT DIVIDE!")


def normalize_array(x, y):
    maxim_x = max(x)
    maxim_y = max(y)
    if maxim_x == 0.0:
        maxim_x = maxim_y
    elif maxim_y == 0.0:
        maxim_y = maxim_x
    for element in x:
        verify_divide(maxim_x)
        element = element / maxim_x
    for element in y:
        verify_divide(maxim_y)
        element = element / maxim_y
    return x, y


def calculate_norm_vectors(x, y):
    n = len(x)
    sum1 = 0
    # x, y = normalize_array(x, y)
    for i in range(n):
        sum1 += abs(x[i] - y[i])
    norm = math.sqrt(sum1)
    return norm


def multiply_matrix_vector(matrix, vector):
    result = []
    n = len(matrix)
    for list_of_lists in matrix:
        dot_product = 0
        for list_ in list_of_lists:
            dot_product += list_[0] * vector[list_[1]]
        result.append(dot_product)
    return result


def verify_matrices_equality(matrix1, matrix2):
    n = len(matrix1)
    for i in range(n):
        for list1 in matrix1[i]:
            for list2 in matrix2[i]:
                if list1[1] == list2[1]:
                    if abs(list1[0] - list2[0]) >= 10 ** (-6):
                        return False
    return True


def read_matrix_only(file1):
    lines = []
    with open(file1, "r") as file_in:
        for line in file_in:
            lines.append(line)
    n = int(lines[0])
    a = [[] for i in range(n)]
    for line in lines:
        elements = line.split(",")
        if len(elements) > 1:
            number = float(elements[0].strip())
            i = int(elements[1].strip())
            j = int(elements[2].strip())
            my_list = [number, j]
            if len(a[i]) != 0:
                ok = False
                for k in range(0, len(a[i])):
                    element_list = a[i][k]
                    if element_list[1] == my_list[1]:
                        print(element_list[0], my_list[0])
                        suma = element_list[0] + my_list[0]
                        a[i][k] = []
                        a[i][k] = [suma, j]
                        ok = True
                        break
                if ok is False:
                    a[i].append(my_list)
                new_list = sorted(a[i], key=lambda x: x[1])
                a[i] = []
                a[i] = [elem for elem in new_list]
            else:
                a[i].append(my_list)
    return n, a


def read_system(file1, file2):
    b = []
    lines = []
    with open(file1, "r") as file_in:
        for line in file_in:
            lines.append(line)
    n = int(lines[0])
    a = [[] for i in range(n)]
    for line in lines:
        elements = line.split(",")
        if len(elements) > 1:
            number = float(elements[0].strip())
            i = int(elements[1].strip())
            j = int(elements[2].strip())
            my_list = [number, j]
            if len(a[i]) != 0:
                ok = False
                for k in range(0, len(a[i])):
                    element_list = a[i][k]
                    if element_list[1] == my_list[1]:
                        print(element_list[0], my_list[0])
                        suma = element_list[0] + my_list[0]
                        a[i][k] = []
                        a[i][k] = [suma, j]
                        ok = True
                        break
                if ok is False:
                    a[i].append(my_list)
                new_list = sorted(a[i], key=lambda x: x[1])
                a[i] = []
                a[i] = [elem for elem in new_list]
            else:
                a[i].append(my_list)

    lines2 = []
    with open(file2, "r") as file_in:
        for line in file_in:
            lines2.append(line)
    for index in range(0, n):
        number = float(lines2[index])
        b.append(number)
    return n, a, b


def verify_null_on_diagonals(a, n):
    for index in range(1, n):
        flag1 = 0
        for element_tuple in a[index]:
            if element_tuple[1] == index:
                if element_tuple[0] != 0:
                    flag1 = 1
        if flag1 == 0:
            return False
    return True


def solve_gauss_seidel(a, b, n):
    epsilon = 1e-6
    k_max = 10000
    k = 0
    norm = 0
    # xc = [0.0 for i in range(n)]
    # xp = [0.0 for i in range(n)]
    xc = [1.0, 2.0, 3.0, 4.0, 5.0]
    xp = [elem for elem in xc]
    for i in range(0, n):
        sum1 = 0.0
        sum2 = 0.0
        element_diagonal = 0.0
        for element_tuple in a[i]:
            if element_tuple[1] < i:
                sum1 += element_tuple[0] * xc[element_tuple[1]]
            elif element_tuple[1] == i:
                element_diagonal = element_tuple[0]
            elif element_tuple[1] > i:
                sum2 += element_tuple[0] * xp[element_tuple[1]]
        verify_divide(element_diagonal)
        xc[i] = (b[i] - sum1 - sum2) / element_diagonal
    norm = calculate_norm_vectors(xc, xp)
    k += 1
    print("k, xc", k, xc)

    while epsilon <= norm <= 10 ** 8 and k <= k_max:
        xp = [elem for elem in xc]
        print(xp)
        for i in range(0, n):
            sum1 = 0.0
            sum2 = 0.0
            element_diagonal = 0.0
            for element_tuple in a[i]:
                if element_tuple[1] < i:
                    sum1 += element_tuple[0] * xc[element_tuple[1]]
                elif element_tuple[1] == i:
                    element_diagonal = element_tuple[0]
                elif element_tuple[1] > i:
                    sum2 += element_tuple[0] * xp[element_tuple[1]]
            verify_divide(element_diagonal)
            xc[i] = (b[i] - sum1 - sum2) / element_diagonal
        norm = calculate_norm_vectors(xc, xp)
        k += 1
        print("k, xc", k, xc)

    print("k=", k)
    if norm < epsilon:
        return xc
    else:
        return "DIVERGENTA"


def add_matrices_bonus(matrix1, matrix2):
    n = len(matrix1)
    new_matrix = [[] for i in range(n)]
    for i in range(n):
        for list_1 in matrix1[i]:
            flag = False
            for list_2 in matrix2[i]:
                if list_1[1] == list_2[1]:
                    flag = True
                    new_element = [list_1[0] + list_2[0], list_1[1]]
                    new_matrix[i].append(new_element)
            if flag is False:
                new_matrix[i].append(list_1)
    for i in range(n):
        for list_1 in matrix2[i]:
            flag = False
            for list_2 in new_matrix[i]:
                if list_1[1] == list_2[1]:
                    flag = True
            if flag is False:
                new_matrix[i].append(list_1)
    return new_matrix


def solve(file1, file2):
    n, a, b = read_system(file1, file2)
    print(a)
    print("The matrix has all elements on the diagonals !=0 :", verify_null_on_diagonals(a, n))
    if verify_null_on_diagonals(a, n):
        xc = solve_gauss_seidel(a, b, n)

        if xc == "DIVERGENTA":
            print("DIVERGENTA")
        else:
            print("x=", xc)
            a_xc = multiply_matrix_vector(a, xc)
            norm = calculate_norm_vectors(a_xc, b)
            print("norm:", norm)
    else:
        print("It cannot be solved")


if __name__ == '__main__':
    print('------a6 b6-------')
    solve("data/a_2.txt", "data/b_2.txt")

    '''------BONUS-------'''
    n1, a1 = read_matrix_only("data/a.txt")
    n2, a2 = read_matrix_only("data/b.txt")
    n3, a3 = read_matrix_only("data/aplusb.txt")

    bonus_matrix = add_matrices_bonus(a1, a2)
    print("----BONUS-----")
    if verify_matrices_equality(bonus_matrix, a3) is True:
        print("The matrices are equal")
    else:
        print("The matrices are not equal")