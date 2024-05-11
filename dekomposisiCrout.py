import numpy as np

def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for j in range(n):
        U[j][j] = 1
        
        for i in range(j, n):
            sum = 0
            for k in range(j):
                sum += L[i][k] * U[k][j]
            L[i][j] = A[i][j] - sum
            
        for i in range(j+1, n):
            sum = 0
            for k in range(j):
                sum += L[j][k] * U[k][i]
            U[j][i] = (A[j][i] - sum) / L[j][j]
    
    return L, U

def solve_linear_eq_crout(A, b):
    L, U = crout_decomposition(A)
    n = len(A)
    
    # Langkah 1: Mencari solusi dari Ly=b dengan substitusi maju
    y = np.zeros(n)
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i][j] * y[j]
        y[i] = (b[i] - sum) / L[i][i]
    
    # Langkah 2: Mencari solusi dari Ux=y dengan substitusi mundur
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum = 0
        for j in range(i+1, n):
            sum += U[i][j] * x[j]
        x[i] = (y[i] - sum) / U[i][i]
    
    return x

# Contoh penggunaan
A = np.array([[3, 2, -1],
              [2, -2, 4],
              [-1, 0.5, -1]])
b = np.array([1, -2, 0])

x = solve_linear_eq_crout(A, b)
print("Solusi x:", x)





#Testing Code
def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for j in range(n):
        U[j][j] = 1
        
        for i in range(j, n):
            sum = 0
            for k in range(j):
                sum += L[i][k] * U[k][j]
            L[i][j] = A[i][j] - sum
            
        for i in range(j+1, n):
            sum = 0
            for k in range(j):
                sum += L[j][k] * U[k][i]
            U[j][i] = (A[j][i] - sum) / L[j][j]
    
    return L, U

def solve_linear_eq_crout(A, b):
    L, U = crout_decomposition(A)
    n = len(A)
    
    # Langkah 1: Mencari solusi dari Ly=b dengan substitusi maju
    y = np.zeros(n)
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i][j] * y[j]
        y[i] = (b[i] - sum) / L[i][i]
    
    # Langkah 2: Mencari solusi dari Ux=y dengan substitusi mundur
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum = 0
        for j in range(i+1, n):
            sum += U[i][j] * x[j]
        x[i] = (y[i] - sum) / U[i][i]
    
    return x

# Contoh penggunaan
A = np.array([[3, 2, -1],
              [2, -2, 4],
              [-1, 0.5, -1]])
b = np.array([1, -2, 0])

# Menggunakan solusi dari numpy sebagai solusi yang diharapkan
expected_solution = np.linalg.solve(A, b)

# Menggunakan fungsi solve_linear_eq_crout untuk mendapatkan solusi
computed_solution = solve_linear_eq_crout(A, b)

# Membandingkan hasil dengan toleransi kecil (epsilon)
epsilon = 1e-8
if np.allclose(computed_solution, expected_solution, atol=epsilon):
    print("Testing passed! Solusi yang dihasilkan sesuai dengan solusi yang diharapkan.")
else:
    print("Testing failed! Solusi yang dihasilkan tidak sesuai dengan solusi yang diharapkan.")

