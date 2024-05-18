import numpy as np
import math


def getb(A,s):
    b=A.dot(s)
    return b

def getQR(A,eps):
    #print(A.shape)
    n = A.shape[0]
    #Q = np.zeros_like(A, dtype=np.float64)
    #R = np.zeros_like(A, dtype=np.float64)
    Q = np.eye(n)
    #print(Q)
    #print(A)
    for r in range(n-1):
        #r=i-1
        P=np.eye(n)
        sigma=0
        for j in range(r,n):
            sigma+=A[j][r]*A[j][r]
        if sigma<eps:
            break
        #print("sigma "+str(sigma))
        k=math.sqrt(sigma)
        if A[r][r]>0:
            k=-k
        #print("k")
        #print(k)
        beta=sigma-k*A[r][r]
        #print("beta "+str(beta))
        if beta==0:
            print("Erooooooooooooor")
        u=[]
        cnt=0
        #print("u shape")
        for j in range(0,r):
            u.append(0)
            cnt+=1
            #print("Contor: "+str(cnt))
            #u[j]=0
        u.append(A[r][r]-k)
        #print("ur")
        #print(A[r][r]-k)
        cnt+=1
        #print("Contor: "+str(cnt))
        #print(len(u))
        #u[r-1]=A[r][r]-k
        for j in range(r+1,n):
            u.append(A[j][r])
            #print("app")
            #print(A[j][r])
            cnt+=1
            #print("Contor: "+str(cnt))
            #u[j]=A[j][r]
        #print("u")
        #print(u)
        V=np.outer(u,u)
        P-=1/beta*V
        A=np.matmul(P,A)
        Q=np.matmul(P,Q)
        #print("A")
        #print(Q)
        #print(A)
        #print("V")
        #print(V)
    Q=Q.transpose()
    for i in range(n):
        if abs(A[i][i])<=eps:
            print("Eroare A singulara")
            exit(1)
    return Q,A

# din tema 2
def substit_inversa(U,y):
    n=U.shape[0]
    x = np.zeros_like(y, dtype=np.float64)
    for i in range(n-1,-1,-1):
        sum=0
        for k in range(i+1,n):
            sum+=U[i,k]*x[k]
        x[i]=(y[i]-sum)/U[i,i]
    return x

def inv_qr(A,eps):
    n=A.shape[0]
    Ainv = np.zeros_like(A, dtype=np.float64)
    Q,R=getQR(A,eps)
    for i in range(n):
        Ainv[:,i] = substit_inversa(R,Q[i,:])
    return Ainv

def produs_rq_upper(R,Q):
    n=R.shape[0]
    A=np.zeros_like(R,dtype=np.float64)
    for i in range(n):
        for j in range(n):
            ans=0
            for k in range(i,n):
                ans+=R[i][k]*Q[k][j]
            A[i][j]=ans
    return A

def bonus(A,eps):
    max_iter=1000
    k=0
    A_k=A.copy()
    #print(A_k.shape)
    while 1:
        #print(A_k.shape)
        Q,R=getQR(A_k,eps)
        #A_nxt=R.dot(Q)
        A_nxt=produs_rq_upper(R,Q)
        #print(A_nxt)
        #print(A_nxt2)
        #print(np.allclose(A_nxt,A_nxt2))
        err=np.linalg.norm(A_k-A_nxt)
        A_k=A_nxt
        #print(A_k.shape)
        k+=1
        if err<=eps or k>=max_iter:
            break
    print("Bonus:")
    print(A_k)
    print(k)
    print(err)
    
    

if __name__ == "__main__":
    A = np.array([[0,0,4],[1,2,3],[0,1,2]], dtype=np.float64)
    #A = np.array([[0,0,4],[0,0,4],[0,1,2]], dtype=np.float64)
    #A = np.array([[1,3,5],[1,3,1],[2,-1,7]], dtype=np.float64)
    s = np.array([3,2,1],dtype=np.float64)
    eps= math.pow(10,-6)
    b=getb(A,s)
    print("Ex1:b")
    print(b)
    Q,R=getQR(A,eps)
    print("Ex2:Q and R")
    print(Q)
    print(R)
    x_householder=substit_inversa(R, np.transpose(Q).dot(b))
    Qlib,Rlib=np.linalg.qr(A)
    x_qr=substit_inversa(Rlib,np.transpose(Qlib).dot(b))
    print("Ex3:norma")
    #print(x_householder)
    #print(x_qr)
    norma=np.linalg.norm(x_householder-x_qr)
    print(norma)
    print("Ex4:normele euclidiene")
    norma1=np.linalg.norm(A.dot(x_householder)-b)
    print(norma1)
    norma2=np.linalg.norm(A.dot(x_qr)-b)
    print(norma2)
    divnorm1=np.linalg.norm(x_householder-s)/np.linalg.norm(s)
    print(divnorm1)
    divnorm2=np.linalg.norm(x_qr-s)/np.linalg.norm(s)
    print(divnorm2)
    A_householder=inv_qr(A,eps)
    A_bibl=np.linalg.inv(A)
    print("Ex5:norma inv")
    normainv=np.linalg.norm(A_householder-A_bibl)
    print(normainv)
    #bonus
    A = np.array([[7,3,0],[3,9,2],[0,2,9]], dtype=np.float64)
    print("deta")
    print(np.linalg.det(A))
    #print(A.shape)
    bonus(A,eps)
    
    
    