#Ery Utami (21120122130054)

import numpy as np
import unittest

def dekomposisi_lu(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1

    for k in range(n):
        U[k, k:] = A[k, k:]
        L[k+1:, k] = A[k+1:, k] / U[k, k]
        A[k+1:, k+1:] -= np.outer(L[k+1:, k], U[k, k+1:])

    return L, U

def lu_gauss(A):
    L, U = dekomposisi_lu(A)
    return L, U

#contoh implementasi

#menetapkan tipe data float
x = np.array([[2, 3, -1], [4, 4, -3], [-2, 3, -1]], dtype=float) 
L, U = lu_gauss(x)
print("Solusi SPL dengan Metode Dekomposisi LU Gauss:")
print("L:", L)
print("U:", U)

class TesDekomposisiLU(unittest.TestCase):
    def tes_dekomposisi(self):
        A = np.array([[2, 3, -1], [4, 4, -3], [-2, 3, -1]])
        L, U = lu_gauss(A)
        np.testing.assert_allclose(np.dot(L, U), A, atol=1e-10)

if __name__ == '__main__':
    unittest.main()