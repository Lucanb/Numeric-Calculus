def sum_equal(u):
    if u+1 == 1:
        return False
    else:
        return True
    
if __name__ == "__main__":
    u = 1
    m = 0
    while sum_equal(u):
        m += 1
        d = u
        u = u/10
        if sum_equal(u) is not True:
            print(f"u-ul meu este : {d} obtinut pentru putearea mea m = {m}")

