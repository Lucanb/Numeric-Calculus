import numpy as np
import urllib.request
import time

# Funcția care încarcă matricea rară din fișierul dat prin link
def load_sparse_matrix_from_link(link, epsilon=1e-6):
    try:
        content_a = urllib.request.urlopen(link).read().decode('utf-8').split('\n')
        n = int(content_a[0])
        valori = []
        ind_col = []
        inceput_linii = [0]

        for line in content_a[1:]:
            val = line.split(',')
            if len(val) != 3:
                continue
            x = float(val[0].strip())
            if abs(x) < epsilon:
                x = 0  # Valorile mai mici decât epsilon sunt considerate 0
            i = int(val[1].strip())
            j = int(val[2].strip())
            valori.append(x)
            ind_col.append(j)
            if i > len(inceput_linii) - 1:
                inceput_linii.append(len(valori))
        return np.array(valori), np.array(ind_col), np.array(inceput_linii)
    except Exception as e:
        print("Eroare la încărcarea matricei rare:", e)
        return None, None, None

# Funcția care încarcă datele pentru sistemul liniar din fișierele a și b pentru un anumit index
def load_system(file_index):   #asta o facem cu compresed lines
    try:
        base_url = "https://profs.info.uaic.ro/~ancai/CN/lab/4/sislinrar/"
        file_a = f"{base_url}a_{file_index}.txt"
        file_b = f"{base_url}b_{file_index}.txt"
        
        valori, ind_col, inceput_linii = load_sparse_matrix_from_link(file_a)
        content_b = urllib.request.urlopen(file_b).read().decode('utf-8').split('\n')
        b_vector = [float(val) for val in content_b[1:] if len(val.strip()) > 0]
        
        return valori, ind_col, inceput_linii, b_vector
    except Exception as e:
        print(f"Eroare la încărcarea sistemului liniar pentru indexul {file_index}:", e)
        return None, None, None, None
    
def load_system_bcrs(file_index, block_size):
    try:
        valori, ind_col, inceput_linii, b = load_system(file_index)
        if valori is None or ind_col is None or inceput_linii is None or b is None:
            return None, None, None, None

        n = len(inceput_linii) - 1
        block_count = n // block_size
        block_dimension = block_size ** 2
        
        val_bcrs = np.zeros((block_count, block_dimension))
        col_ind_bcrs = np.zeros(block_count * block_dimension, dtype=int)
        row_blk_bcrs = np.zeros(block_count + 1, dtype=int)

        for block_index in range(block_count):
            row_blk_bcrs[block_index] = block_index * block_dimension
            for i in range(block_size):
                for j in range(block_size):
                    val_bcrs[block_index, i * block_size + j] = valori[inceput_linii[block_index] + i * n + j]
                    col_ind_bcrs[block_index * block_dimension + i * block_size + j] = ind_col[inceput_linii[block_index] + i * n + j]
        row_blk_bcrs[block_count] = block_count * block_dimension

        return val_bcrs, col_ind_bcrs, row_blk_bcrs, b
    except Exception as e:
        print(f"Eroare la încărcarea sistemului liniar în format BCRS pentru indexul {file_index}:", e)
        return None, None, None, None    
# vezi    
def gauss_seidel_sparse(A, b, inceput_linii=None, epsilon=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    iter_count = 0
    
    start_time = time.time()
    
    while iter_count < max_iter:
        for i in range(n):
            if inceput_linii is not None:  # Verificați dacă există indici de început al liniilor
                start_idx = inceput_linii[i]
                end_idx = inceput_linii[i + 1] if i + 1 < len(inceput_linii) else len(A)
                sum1 = np.dot(A[start_idx:end_idx], x_new)
                sum2 = np.dot(A[start_idx:end_idx, i + 1:], x[i + 1:])
                x_new[i] = (b[i] - sum1 - sum2) / A[start_idx, i]
            else:
                sum1 = np.dot(A[i], x_new)
                sum2 = np.dot(A[i + 1:], x[i + 1:])
                x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
            
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            break
        
        x = np.copy(x_new)
        iter_count += 1
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return x, iter_count, execution_time

# Iterăm peste toate indexurile de la 1 la 5 și încărcăm sistemele liniare
for i in range(1, 6):
#test meth 1
    # print(f"\nÎncărcăm sistemul liniar cu indexul {i}:")
    # valori, ind_col, inceput_linii, b = load_system(i)
    # if valori is not None and ind_col is not None and inceput_linii is not None and b is not None:
    #     print("Valori nenule ale matricei rare:")
    #     print(valori)
    #     print("Indicii de coloană ai matricei rare:")
    #     print(ind_col)
    #     print("Vectorul de început al liniilor:")
    #     print(inceput_linii)
    #     print("Vectorul termenilor liberi b:")
    #     print(b)
    # else:
    #     print("Nu s-a putut încărca sistemul liniar.")
#test meth 2
    # print(f"\nÎncărcăm sistemul liniar cu indexul {i} în formatul BCRS:")
    # block_size = 3  # Dimensiunea blocului pentru BCRS (poate fi modificată)
    # val_bcrs, col_ind_bcrs, row_blk_bcrs, b_bcrs = load_system_bcrs(i, block_size)
    # if val_bcrs is not None and col_ind_bcrs is not None and row_blk_bcrs is not None and b_bcrs is not None:
    #     print("Valori blocuri matrice rară (BCRS):")
    #     print(val_bcrs)
    #     print("Indici coloană blocuri matrice rară (BCRS):")
    #     print(col_ind_bcrs)
    #     print("Pointeri început blocuri pe linii (BCRS):")
    #     print(row_blk_bcrs)
    #     print("Vectorul termenilor liberi b:")
    #     print(b_bcrs)
    # else:
    #     print("Nu s-a putut încărca sistemul liniar în formatul BCRS.")
    #     exit()

    print(f"\nAproximăm soluția sistemului liniar cu indexul {i} folosind metoda Gauss-Seidel:")
    for method in ['CSR', 'BCRS']:
        print(f"\nMetoda de memorare rară: {method}")
        if method == 'CSR':
            valori, ind_col, inceput_linii, b = load_system(i)
            if valori is not None and ind_col is not None and inceput_linii is not None and b is not None:
                solutie_aproximativa, numar_iteratii, timp_executie = gauss_seidel_sparse(valori, b, inceput_linii)
                print("Soluție aproximativă:")
                print(solutie_aproximativa)
                print("Numărul de iterații necesare:")
                print(numar_iteratii)
                print("Timpul de execuție (secunde):")
                print(timp_executie)
                
                # Calcularea soluției exacte cu Gauss-Seidel
                solutie_exacta, _, _ = gauss_seidel_sparse(valori, b, inceput_linii, epsilon=1e-15)
                
                # Calcularea normei diferenței dintre soluția aproximativă și cea exactă
                norma_diferenta = np.linalg.norm(solutie_aproximativa - solutie_exacta, ord=np.inf)
                print("Norma diferenței între soluția aproximativă și cea exactă:")
                print(norma_diferenta)
            else:
                print("Nu s-a putut încărca sistemul liniar.")
        else:
            block_size = 3
            valori, ind_col, inceput_linii, b = load_system_bcrs(i,block_size)
            if valori is not None and ind_col is not None and inceput_linii is not None and b is not None:
                solutie_aproximativa, numar_iteratii, timp_executie = gauss_seidel_sparse(valori, b, inceput_linii)
                print("Soluție aproximativă:")
                print(solutie_aproximativa)
                print("Numărul de iterații necesare:")
                print(numar_iteratii)
                print("Timpul de execuție (secunde):")
                print(timp_executie)
                
                # Calcularea soluției exacte cu Gauss-Seidel
                solutie_exacta, _, _ = gauss_seidel_sparse(valori, b, inceput_linii, epsilon=1e-15)
                
                # Calcularea normei diferenței dintre soluția aproximativă și cea exactă
                norma_diferenta = np.linalg.norm(solutie_aproximativa - solutie_exacta, ord=np.inf)
                print("Norma diferenței între soluția aproximativă și cea exactă:")
                print(norma_diferenta)
            else:
                print("Nu s-a putut încărca sistemul liniar.")
