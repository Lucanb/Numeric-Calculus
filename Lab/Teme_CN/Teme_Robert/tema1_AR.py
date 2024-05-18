import random
import math

def subpct1():
    ans=10
    while True:
        nans=ans/10
        if (1+nans==1):
            print(ans)
            return ans
        else:
            ans=nans
            #print(ans)
    
def subpct2():
    x=1.0
    u=subpct1()
    y=u/10
    z=u/10
    if (x+y)+z!=x+(y+z):
        print("Diferit")
    else:
        print("Egal")

def subpct2p2():
    u=subpct1()
    x=10**30
    y=u
    z=u
    if (x*y)*z!=x*(y*z):
        print("Diferit")
    else:
        print("Egal")
        
def T(i, a):
    ans=0
    if i==4:
        ans=105*a-10*(a**3)
        ans/=(105-45*(a**2)+(a**4))
    if i==5:
        ans=945*a-105*(a**3)+(a**5)
        ans/=((945-420*(a**2))+15*(a**4))
    if i==6:
        ans=10395*a-1260*(a**3)+21*(a**5)
        ans/=(10395-4725*(a**2)+210*(a**4)-(a**6))
    if i==7:
        ans=135135*a-17325*(a**3)+378*(a**5)-(a**7)
        ans/=(135135-62370*(a**2)+3150*(a**4)-28*(a**6))
    if i==8:
        ans=2027025*a-270270*(a**3)+6930*(a**5)-36*(a**7)
        ans/=(2027025-945945*(a**2)+51975*(a**4)-630*(a**6)+(a**8))
    if i==9:
        ans=34459425*a-4729725*(a**3)+135135*(a**5)-990*(a**7)+(a**9)
        ans/=(34459425-16216200*(a**2)+945945*(a**4)-13860*(a**6)+45*(a**8))
    return ans
        
def subpct3():
    b=[0]*10
    #print(b)
    for i in range(10000):
        r=random.uniform(-math.pi/2, math.pi/2)
        exact=math.tan(r)
        bst=1e9
        ibst=0
        for j in range(4,10):
            val=T(j,r)
            dif=abs(val-exact)
            if dif<bst:
                bst=dif
                ibst=j
        b[ibst]+=1
    print(b)
 
def S(i,a):
    ans=T(i,a)
    ans/=math.sqrt(1+(T(i,a)**2))
    return ans

def C(i,a):
    ans=1
    ans/=math.sqrt(1+(T(i,a)**2))
    return ans

def bonus():
    print(S(9,math.pi/4))
    print(C(9,math.pi/4))            
    
if __name__ == "__main__":
    #print("Hello world!")
    subpct1()
    subpct2()
    subpct2p2()
    subpct3()
    bonus()