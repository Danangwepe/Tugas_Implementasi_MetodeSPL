import numpy as np

def lu_gauss(A, b):
    n = len(A)
    
    # Inisialisasi matriks L dan U
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Langkah 1: Dekomposisi A menjadi L dan U
    for i in range(n):
        # Matriks U
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = A[i][k] - sum
        
        # Matriks L
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (A[k][i] - sum) / U[i][i]
    
    # Langkah 2: Mencari solusi dari Ly=b dengan substitusi maju
    y = np.zeros(n)
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i][j] * y[j]
        y[i] = (b[i] - sum) / L[i][i]
    
    # Langkah 3: Mencari solusi dari Ux=y dengan substitusi mundur
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

x = lu_gauss(A, b)
print("Solusi x:", x)



#Testing Code
# Fungsi untuk memeriksa apakah dua matriks sama dengan toleransi tertentu
def assert_matrix_equal(m1, m2, tol=1e-6):
    assert np.allclose(m1, m2, atol=tol), "Matrices not equal!"

# Fungsi untuk melakukan pengujian metode dekomposisi LU Gauss
def test_lu_gauss_decomposition():
    # Matriks koefisien A dan vektor hasil b
    A = np.array([[3, 2, -1],
              [2, -2, 4],
              [-1, 0.5, -1]])

    b = np.array([1, -2, 0])
    
    # Solusi yang diharapkan
    expected_solution = np.array([1, -2, -2])
    
    # Metode dekomposisi LU Gauss
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Dekomposisi A menjadi L dan U
    for i in range(n):
        # Matriks U
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = A[i][k] - sum
        
        # Matriks L
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (A[k][i] - sum) / U[i][i]
    
    # Matriks hasil dari perkalian L dan U
    LU_result = np.dot(L, U)
    
    # Memeriksa apakah matriks hasil sama dengan matriks koefisien asli A
    assert_matrix_equal(LU_result, A)
    
    # Memeriksa apakah solusi yang dihasilkan sesuai dengan yang diharapkan
    x = lu_gauss(A, b)  # Perubahan di sini: pemanggilan fungsi yang benar
    assert np.allclose(x, expected_solution), "Incorrect solution!"

# Jalankan pengujian
test_lu_gauss_decomposition()
print("All tests passed successfully!")
