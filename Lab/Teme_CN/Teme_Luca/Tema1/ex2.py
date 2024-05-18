import numpy as np
def sum_equal(u):
    if u+1 == 1:
        return False
    else:
        return True
    
def returnPrecision():
    u = 1
    m = 0
    while sum_equal(u):
        m += 1
        d = u
        u = u/10
        if sum_equal(u) is not True:
            return d

def verifAsociativy():
    x = 1.0
    y = returnPrecision()/10
    z = y

    value1 = (x + y) + z
    value2 = x + (y+z)

    if value1 != value2:
        print(f"Neasociative {x} , {y} , {z}")   
    if value1 == value2:
        print(f"Asociative {x} , {y} , {z}")   

def findNumbers():

    contor = 0
    while True :    
        elements = np.random.rand(3)
        contor +=1
        print(contor)
        value1 = (elements[0]*elements[1])*elements[2]
        value2 = (elements[0]*elements[2])*elements[1]
        if value1 != value2:
            print(f"Neasociative {elements[0]},{elements[1]},{elements[2]}")
            print(f"Value 1 : {value1} Value 2: {value2}")
            break    

if __name__ == "__main__":
    #verifAsociativy()
    findNumbers()