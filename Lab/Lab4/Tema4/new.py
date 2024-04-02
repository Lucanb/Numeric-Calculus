import numpy as np
import urllib.request
import time
from copy import deepcopy

class SparseMatrix:
    def __init__(self,n):
        self.n = n
        self.mat = [{} for _ in range(n)]

    # def __init__(self, n):
    #     self.n = n
    #     self.rows = []
    #     self.cols = []
    #     self.data = []    
    

    def addMyElement(self, i, j, x):
        if x != 0: 
            self.mat[i][j] = self.mat[i].setdefault(j, 0) + x
        
    # def addMyElementBCRS(self, i, j, x):
    #     if x != 0:
    #         self.rows.append(i)
    #         self.cols.append(j)
    #         self.data.append(x)    


    def has_null_diagonal(self):
        for i in range(self.n):
            if i in self.mat[i] and self.mat[i][i] == 0:
                return True
        return False
    
    def getNonZeroValues(self):
        return zip(self.rows, self.cols, self.data)

    # def has_null_diagonalBCRS(self):
    #     diagonal_check = [False] * (self.n)  #mai rapid
    #     for i in range(len(self.rows)):
    #         if self.rows[i] == self.cols[i]:
    #             print(self.rows[i])
    #             diagonal_check[self.rows[i]] = True  # True daca exista

    #     return all(diagonal_check)

    def get_diagonal_value(self, index):
        for i in range(len(self.rows)):
            if self.rows[i] == self.cols[i] == index:  # Verificăm dacă elementul este de pe diagonala principală și are indicele specificat
                return self.data[i]  # Returnăm valoarea elementului de pe diagonala principală
        return None 
    
    def gauss_seidel(self, b, tol=1e-6, max_iter=10000):
        x = [0] * self.n  # Inițializăm vectorul soluție
        # x = [1.0, 2.0, 3.0, 4.0, 5.0]
        k = 0
        delta_x = tol + 1  # Inițializăm delta_x cu o valoare care să satisfacă condiția de intrare în buclă
        while k < max_iter and delta_x > tol:
            xp = x.copy()
            for i in range(self.n):
                sum1 = sum([self.mat[i][j] * x[j] for j in self.mat[i] if j < i])
                sum2 = sum([self.mat[i][j] * xp[j] for j in self.mat[i] if i < j < self.n])
                x[i] = (b[i] - sum1 - sum2) / self.mat[i].get(i, 1)  # Am schimbat 0 cu 1 pentru a evita diviziunea la 0
            delta_x = np.linalg.norm(np.array(x) - np.array(xp))
            k += 1

        if delta_x < tol:
            product = [0] * self.n
            for i in range(self.n):
                for j, val in self.mat[i].items():
                    product[i] += val * x[j]
            norm = np.linalg.norm(np.array(product) - np.array(b), ord=np.inf)
            print('norm(A @ xGS - b) = 0:', norm < tol)
            return x
        else:
            return 'divergence'
        

    def calculate_residual_norm(self, x, b):
        Ax = [sum(self.mat[i].get(j, 0) * x[j] for j in range(self.n)) for i in range(self.n)]

        residual = np.abs(np.array(Ax) - np.array(b))
        norm_inf = np.linalg.norm(residual, np.inf)
            
        return norm_inf  
                      

    # def gauss_seidelBCRS(self, b, tol=1e-6, max_iter=10000):
    #     # xc = np.zeros(self.n)  # Inițializăm vectorul soluție cu zero
    #     xp = np.zeros(self.n)  # Vectorul anterior
    #     xc = [1.0, 2.0, 3.0, 4.0, 5.0]
    #     k = 0
    #     delta_x = tol + 1  # Inițializăm delta_x cu o valoare care să satisfacă condiția de intrare în buclă
    #     while k < max_iter and delta_x >= tol and delta_x <= 1e8:  # Condiția pentru delta_x trebuie să fie >= tol și <= 1e8 conform cerințelor
    #         xp = np.copy(xc)
    #         # print(len(self.rows))
    #         for i in range(self.n):
    #             value = None
    #             sum1 = 0
    #             sum2 = 0
    #             print('rows : ',self.rows)
    #             print('cols : ',self.cols)
    #             print('values : ',self.data)
    #             for j in range(len(self.rows)):
    #                 if self.rows[j] == i and self.cols[j] < i:
    #                     sum1 += self.data[j] * xc[self.cols[j]]
    #                     print('j : ',i,self.cols[j])
    #                     print('elements : ',self.data[j],xc[self.cols[j]])
    #                     print('sum1 : ',sum1)
    #                 elif self.rows[j] == i and self.cols[j] == i :
    #                     value = self.data[j]
    #                     # print(value,i,j)    
    #                 elif self.rows[j] == i and self.cols[j] > i:
    #                     sum2 += self.data[j] * xp[self.cols[j]]
    #                     print('sum2 : ',sum2)   
    #             xc[i] = (b[i] - sum1 - sum2) / value 
    #         delta_x = np.linalg.norm(xc - xp)
    #         k += 1

    #     print(xc)
    #     if delta_x < tol:
    #         return xc
    #     else:
    #         return 'divergence'


    # def calculate_residual_normBCRS(self, x, b):
    #     residual = np.zeros(5)
    #     for i, j, val in self.getNonZeroValues():
    #         residual[i] += val * x[j]
    #     residual = np.abs(residual - b)
    #     norm_inf = np.linalg.norm(residual, np.inf)
    #     return norm_inf 

def loadData(index):
    try:
        with open(f"data/a_{index}.txt",'r') as file:
            n1 = int(file.readline().strip())
            a = SparseMatrix(n1)
            for line in file:
                line = line.strip()
                if line:
                    val = line.split(',')
                    if all(val):                     # aici am verificat ca sa am pereu triplet nevid   
                        if len(val) != 3:
                            print(f"Eroare: Linia '{line}' nu conține trei elemente separate de virgulă.")
                            continue
                        x = float(val[0].strip())
                        i = int(val[1].strip())
                        j = int(val[2].strip())
                        a.addMyElement(i, j, x)
                    else:
                        continue    
                else:
                    continue

        with open(f"data/b_{index}.txt",'r') as file:
            n2 = int(file.readline().strip())
            b = []
            
            for _ in range(n2):
                line = file.readline().strip()
                if line:
                    b_val = float(line)
                    b.append(b_val)
                else:
                    continue

        return n1,n2,a,b
  
    except Exception as e:
        print(f"Eroare la incarcarea fisierului {index}:", e)
        return None, None, None, None

# def loadBCRS(index):
#     try:
#         with open(f"data/a_{index}.txt", 'r') as file:
#             n = int(file.readline().strip())
#             a = SparseMatrix(n)
#             for line in file:
#                 line = line.strip()
#                 if line:
#                     val = line.split(',')
#                     if all(val):  # Verificăm dacă avem toate elementele necesare
#                         if len(val) != 3:
#                             print(f"Eroare: Linia '{line}' nu conține trei elemente separate de virgulă.")
#                             continue
#                         x = float(val[0].strip())
#                         i = int(val[1].strip())
#                         j = int(val[2].strip())
#                         a.addMyElementBCRS(i, j, x)
#                     else:
#                         continue
#                 else:
#                     continue

#         with open(f"data/b_{index}.txt", 'r') as file:
#             n2 = int(file.readline().strip())
#             b = []
#             for _ in range(n2):
#                 line = file.readline().strip()
#                 if line:
#                     b_val = float(line)
#                     b.append(b_val)
#                 else:
#                     continue

#         return n, n2, a, b

#     except Exception as e:
#         print(f"Eroare la încărcarea fișierului {index}:", e)
#         return None, None, None, None

def sumMat():
        n1,n2,n3,a,b,s = loadBonus()
        print(a.mat[0][0])
        print(b.mat[0][0])
        print(s.mat[0].get(0) == None)
        for i in range(n1):
            for j in range(n2):  # Adăugați această linie pentru a defini j în cadrul acestui bloc
                if (a.mat[i].get(j) and b.mat[i].get(j) and not((a.mat[i].get(j) + b.mat[i].get(j) == 0 and s.mat[i].get(j) is None))):
                        if(a.mat[i].get(j) and b.mat[i].get(j) and not(a.mat[i].get(j) + b.mat[i].get(j) == s.mat[i].get(j)) ):
                            print(f"Elementele care îndeplinesc condiția: a[{i}][{j}] = {a.mat[i][j]}, s[{i}][{j}] = {s.mat[i].get(j)}")
                            return False
        return True 

def loadBonus():
    try:
        with open(f"data/a.txt",'r') as file:
            n1 = int(file.readline().strip())
            a = SparseMatrix(n1)
            for line in file:
                line = line.strip()
                if line:
                    val = line.split(',')
                    if all(val):                     # aici am verificat ca sa am pereu triplet nevid   
                        if len(val) != 3:
                            print(f"Eroare: Linia '{line}' nu conține trei elemente separate de virgulă.")
                            continue
                        x = float(val[0].strip())
                        i = int(val[1].strip())
                        j = int(val[2].strip())
                        a.addMyElement(i, j, x)
                    else:
                        continue    
                else:
                    continue
        
        with open(f"data/aplusb.txt",'r') as file:
            n3 = int(file.readline().strip())
            s = SparseMatrix(n1)
            for line in file:
                line = line.strip()
                if line:
                    val = line.split(',')
                    if all(val):                     # aici am verificat ca sa am pereu triplet nevid   
                        if len(val) != 3:
                            print(f"Eroare: Linia '{line}' nu conține trei elemente separate de virgulă.")
                            continue
                        x = float(val[0].strip())
                        i = int(val[1].strip())
                        j = int(val[2].strip())
                        s.addMyElement(i, j, x)
                    else:
                        continue    
                else:
                    continue                

        with open(f"data/b.txt",'r') as file:
            n2 = int(file.readline().strip())
            b = SparseMatrix(n1)
            for line in file:
                line = line.strip()
                if line:
                    val = line.split(',')
                    if all(val):                     # aici am verificat ca sa am pereu triplet nevid   
                        if len(val) != 3:
                            print(f"Eroare: Linia '{line}' nu conține trei elemente separate de virgulă.")
                            continue
                        x = float(val[0].strip())
                        i = int(val[1].strip())
                        j = int(val[2].strip())
                        b.addMyElement(i, j, x)
                    else:
                        continue    
                else:
                    continue 

        return n1,n2,n3,a,b,s
  
    except Exception as e:
        print(f"Eroare la incarcarea fisierului :", e)
        return None, None, None, None


# n1, n2, a, b = loadData(1)
# if a.has_null_diagonal():
#     print("Matricea are un element diagonal nul. Sistemul nu poate fi rezolvat folosind metoda iterativă Gauss-Seidel.")
# else:
#     print('e oke')
#     x_approx = a.gauss_seidel(b)
#     if isinstance(x_approx, list):  # Verificăm dacă x_approx este un vector numeric
#         print("Soluția aproximată:", x_approx)
#         residual_norm = a.calculate_residual_norm(x_approx, b)
#         print("Norma reziduului:", residual_norm)
#     else:
#         print("Algoritmul Gauss-Seidel diverge.")


# # Definirea unei funcții pentru a crea fișierele de intrare a_1.txt și b_1.txt
# def create_input_files(A_data, b_data):
#     with open("data/a_0.txt", "w") as f:
#         for row in A_data:
#             f.write(" ".join(map(str, row)) + "\n")

#     with open("data/b_0.txt", "w") as f:
#         f.write("\n".join(map(str, b_data)))

# # Crearea fișierelor de intrare
# create_input_files(A_data, b_data)


# n1, n2, a, b = loadBCRS(0)
# if  a.has_null_diagonalBCRS(): 
#         print('e oke')
#         x_approx = a.gauss_seidelBCRS(b)
#         if isinstance(x_approx, list):  # Verificăm dacă x_approx este un vector numeric
#             print("Soluția aproximată:", x_approx)
#             residual_norm = a.calculate_residual_normBCRS(x_approx, b)
#             print("Norma reziduului:", residual_norm)
#         else:
#             print("Algoritmul Gauss-Seidel diverge.")
# else:
#     print('Nu are solutie')      

#[-1329.26585366  8266.53658537 -3473.75498495 ...   244.7804878
  # 250.82926829   247.80487805]    

print(sumMat())