import numpy as np

eps = 1e-6

def householderAgo(a_init, b_init=None):
    a = a_init.copy()
    if b_init is not None:
        b = b_init.copy()
    n = len(a)
    q_tilda = np.identity(n)
    u = np.empty(n)
    for r in range(n - 1):
        sigma = sum([a[i][r] ** 2 for i in range(r, n)])
        if sigma <= eps:
            break
        k = np.sqrt(sigma)
        if a[r][r] > 0:
            k *= -1
        beta = sigma - k * a[r][r]
        u[r] = a[r][r] - k
        for i in range(r + 1, n):
            u[i] = a[i][r]
        for j in range(r + 1, n):
            gamma = sum([u[i] * a[i][j] for i in range(r, n)]) / beta
            for i in range(r, n):
                a[i][j] -= gamma * u[i]
        a[r][r] = k
        for i in range(r + 1, n):
            a[i][r] = 0
        if b_init is not None:
            gamma = sum([u[i] * b[i] for i in range(r, n)]) / beta
            for i in range(r, n):
                b[i] -= gamma * u[i]
        for j in range(n):
            gamma = sum([u[i] * q_tilda[i][j] for i in range(r, n)]) / beta
            for i in range(r, n):
                q_tilda[i][j] -= gamma * u[i]
    q = np.transpose(q_tilda)
    r = a.copy()
    return q, r


def inverse_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < eps:
            print("sistem fara solutie")
            return None
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]

    return x

def calculate_norms(a, b, x, s):
    norm1 = np.linalg.norm(a @ x - b)
    norm2 = np.linalg.norm(x - s) / np.linalg.norm(s)
    if norm1 >= eps and norm2 >= eps:
        return None,None
    else:
        return norm1,norm2
     
def calculateInvQR(q, r):
    n = len(q)
    inv = np.empty((n, n))
    for j in range(n):
        inv[:, j] = inverse_substitution(r, q[j, :])
    return inv

def randomSystemValues(n): #Ex 6 - random Initialisator
    a = np.random.uniform(-99, 100, (n, n))
    s = np.random.uniform(-99, 100, n)
    b = a @ s
    return a, b, s

def generateSymetricMat(n):   # for Bonus
    a = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            a[i][j] = a[j][i] = np.random.uniform(-99, 100)
    return a

def calculateLimit(a):
    q, r = householderAgo(a)
    while True:
        lim = np.linalg.norm(r @ q - q @ r)
        if lim < eps:
            break
        q, r = householderAgo(r @ q)
    print(r @ q)
    return lim

if __name__ == "__main__":
    
    #data initialisation :

    n = int(input("dimensiunea matricilor sistem : "))
    t = int(input("precizia calculelor : "))
    eps = 10 ** (-t)
    a, b, s = randomSystemValues(n)
    
    # a = np.array([
    #     [0, 0, 4],
    #     [1, 2, 3],
    #     [0, 1, 2]
    # ])
    # s = np.array([3, 2, 1])
    # b = a @ s
    
    print('Ex1 :')
    print(f'Vector b = {b}')
    
    print('Ex2 :')
    q, r = householderAgo(a, b)
    print(f'Matrix Q = {q}')
    print(f'Matrix R = {r}')
    
    print('Ex3 :')
    x_householder = inverse_substitution(r, np.transpose(q) @ b)
    new_q,new_r = np.linalg.qr(a) # scot q si r cu libraria
    x_qr = inverse_substitution(new_r, np.transpose(new_q) @ b)
    print(f'x solution with house holder algorithm = {x_householder}')
    print(f'x solution with QR algorithm = {x_qr}')
    norm = np.linalg.norm(x_householder - x_qr) < eps
    print(f'||x_householder - x_qr|| == 0: {norm}')
    
    print('Ex4 :')
    
    norms_hoseholder =  calculate_norms(a, b, x_householder, s)
    norms_x_qr = calculate_norms(a, b, x_qr, s)
    
    if norms_hoseholder != None :
        norm1_householder,norm2_householder = norms_hoseholder
        print(f'Norms for x_householder: {norm1_householder} {norm2_householder}')
    else :
        print(f'Norms for x_householder: False')
    if  norms_x_qr != None :
        norm1_x_qr,norm2_x_qr = norms_x_qr
        print(f'Norms for x_qr: {norm1_x_qr} {norm2_x_qr}')
    else:
        print(f'Norms for x_qr: False')

    print('Ex5 :')
    inv_a_householder = calculateInvQR(q, r)
    abs_inverse = np.linalg.inv(a)
    print(f'inv_a_householder = {inv_a_householder}')
    print(f'inv_A_NP = {abs_inverse}')
    abs_norm_value = np.linalg.norm(inv_a_householder - abs_inverse)
    if  abs_norm_value < eps:
        print('||inv_a_householder - abs_inverse|| == 0: TRUE')
        print('||inv_a_householder - abs_inverse|| :', abs_norm_value)
    else:
        print('||inv_a_householder - abs_inverse|| == 0: FALSE')

    print('Bonus :')
    Mat = generateSymetricMat(n)
    lim = calculateLimit(Mat)
    print('lim =', lim)