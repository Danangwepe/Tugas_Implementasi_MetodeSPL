import numpy as np


# Mendefinisikan matriks koefisien A dan vektor hasil b
A = np.array([[3, 2, -1],
              [2, -2, 4],
              [-1, 0.5, -1]])

b = np.array([1, -2, 0])

# Mencari matriks balikan dari A
A_inv = np.linalg.inv(A)

# Menghitung solusi x
x = np.dot(A_inv, b)

print("Solusi x:", x)



#Testing Code
# Fungsi untuk memeriksa apakah dua array sama dengan toleransi tertentu
def assert_array_equal(arr1, arr2, tol=1e-6):
    assert np.allclose(arr1, arr2, atol=tol), "Arrays not equal!"

# Fungsi untuk melakukan pengujian metode matriks balikan
def test_inverse_matrix_method():
    # Persamaan linear: A*x = b
    A = np.array([[3, 2, -1],
              [2, -2, 4],
              [-1, 0.5, -1]])

    b = np.array([1, -2, 0])
    
    # Solusi yang diharapkan
    expected_solution = np.array([1, -2, -2])
    
    # Metode matriks balikan
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    
    # Memeriksa apakah solusi yang dihasilkan sama dengan yang diharapkan
    assert_array_equal(x, expected_solution)

# Jalankan pengujian
test_inverse_matrix_method()
print("All tests passed successfully!")
