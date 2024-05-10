#Ery Utami (21120122130054)

import numpy as np
import unittest

def matriks_balikan(A):
    #cek apakah punya balikan
    #cari matriks balikan A
    try:
        A_inv = np.linalg.inv(A)
        return A_inv
    except np.linalg.LinAlgError:
        return None

#contoh implementasi
A = np.array([[1, 0, -1], [-2, 3, 0], [1, -3, 2]])      

#solusi
inv_A = matriks_balikan(A)
if inv_A is not None:
    print("Matriks Balikan A:")
    print(inv_A)
else:
    print("Matriks A Tidak Memiliki Balikan.")

class TesMatriksBalikan(unittest.TestCase):
    def tes_matriks_balikan(self):
        # Contoh matriks dengan balikan
        A = np.array([[1, 0, -1], [-2, 3, 0], [1, -3, 2]])
        expected_inv_A = np.array([[2, 1, 1], [1.333333, 1, 0.666667], [1, 1, 1]])

        inv_A = matriks_balikan(A)
        self.assertIsNotNone(inv_A)
        np.testing.assert_array_almost_equal(inv_A, expected_inv_A)

        # Contoh matriks tanpa balikan
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        inv_A = matriks_balikan(A)
        self.assertIsNone(inv_A)

if __name__ == '__main__':
    unittest.main()