import numpy as np

def Crout(A,eps):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.float64)
    U = np.zeros_like(A, dtype=np.float64)

    for p in range(n):
        for i in range(p, n):
            L[i,p] = A[i,p]
            for k in range(p):
                L[i,p]-=L[i,k]*U[k,p]
        for i in range(p+1, n):
            U[p,i]=A[p,i]
            if abs(L[p,p])<eps:
                print("Matricea A este singulara!")
                exit(0)
            for k in range(p):
                U[p,i]-=L[p,k]*U[k,i]
            U[p,i]/=L[p,p]
        for i in range(n):
            U[i,i]=1
    return L, U

def Crout_restrictie(A,eps):
    n = A.shape[0]
    for p in range(n):
        for i in range(p, n):
            for k in range(p):
                A[i,p]-=A[i,k]*U[k,p]
        for i in range(p+1, n):
            if abs(A[p,p])<eps:
                print("Matricea A este singulara!")
                exit(0)
            for k in range(p):
                A[p,i]-=A[p,k]*A[k,i]
            A[p,i]/=A[p,p]

def getl(i,j):
    i+=1
    j+=1
    ans=0
    for t in range(i):
        ans+=t
    ans+=j
    return int(ans)-1

def getu(i,j):
    ans=0
    for t in range(j+1):
        ans+=t
    ans+=i+1
    return int(ans)-1

def bonus(A,eps):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.float64)
    U = np.zeros_like(A, dtype=np.float64)
    Lv = np.zeros(n * (n + 1) // 2, dtype=np.float64)
    Uv = np.zeros(n * (n + 1) // 2, dtype=np.float64)

    for p in range(n):
        for i in range(p, n):
            L[i,p] = A[i,p]
            Lv[getl(i,p)]=A[i,p]
            for k in range(p):
                L[i,p]-=L[i,k]*U[k,p]
                Lv[getl(i,p)]-=Lv[getl(i,k)]*Uv[getu(k,p)]
            if L[i,p]!=Lv[getl(i,p)]:
                print("eroare")
        for i in range(p+1, n):
            Uv[getu(p,i)]=A[p,i]
            U[p,i]=A[p,i]
            if abs(Lv[getl(p,p)])<eps:
                print("Matricea A este singulara!")
                exit(0)
            for k in range(p):
                U[p,i]-=L[p,k]*U[k,i]
                Uv[getu(p,i)]-=Lv[getl(p,k)]*Uv[getu(k,i)]
            U[p,i]/=L[p,p]
            Uv[getu(p,i)]/=Lv[getl(p,p)]
            if U[p,i]!=Uv[getu(p,i)]:
                print("Eroare")
        for i in range(n):
            Uv[getu(i,i)]=1
            U[i,i]=1
    return Lv, Uv
    

def substit_directa(L,b):
    n=L.shape[0]
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        sum=0
        for j in range(i):
            sum+=L[i,j]*y[j]
        y[i]=(b[i]-sum)/L[i,i]
    return y
    
def substit_inversa(U,y):
    n=U.shape[0]
    x = np.zeros_like(y, dtype=np.float64)
    for i in range(n-1,-1,-1):
        sum=0
        for k in range(i+1,n):
            sum+=U[i,k]*x[k]
        x[i]=(y[i]-sum)/U[i,i]
    return x
            
def substit_directa_bonus(L,b,n):
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        sum=0
        for j in range(i):
            sum+=L[getl(i,j)]*y[j]
        y[i]=(b[i]-sum)/L[getl(i,i)]
    return y
    
def substit_inversa_bonus(U,y,n):
    x = np.zeros_like(y, dtype=np.float64)
    for i in range(n-1,-1,-1):
        sum=0
        for k in range(i+1,n):
            sum+=U[getu(i,k)]*x[k]
        x[i]=(y[i]-sum)/U[getu(i,i)]
    return x
            

if __name__ == "__main__":
    A = np.array([[2.5,2,2],[5,6,5],[5,6,6.5]], dtype=np.float64)
    b = np.array([2,2,2],dtype=np.float64)
    n=A.shape[0]
    Ainit=np.array(A)
    t=5
    eps=pow(10,-t)
    print(A)
    L,U=Crout(A,eps)
    print(L)
    print(U)
    sl=L.shape[0]
    su=U.shape[0]
    dl=1
    du=1
    da=1
    for i in range(sl):
        dl*=L[i,i]
    for i in range(su):
        du*=U[i,i]
    da=da*dl*du
    print(da)
    y=substit_directa(L,b)
    x=substit_inversa(U,y)
    print(x)
    bend=Ainit.dot(x)
    norma=np.linalg.norm(bend-b)
    if norma<pow(10,-9):
        print("Solutia e corecta!")
    #print(A)
    Crout_restrictie(A,eps)
    print(A)
    npsolve=np.linalg.solve(Ainit,b)
    nx=np.linalg.norm(x-npsolve)
    print(nx)
    Ainv=np.linalg.inv(Ainit)
    Ainvb=Ainv.dot(b)
    nxinv=np.linalg.norm(x-Ainvb)
    print(nxinv)
    L,U=bonus(Ainit,eps)
    print(L)
    print(U)
    y=substit_directa_bonus(L,b,n)
    x=substit_inversa_bonus(U,y,n)
    print(x)