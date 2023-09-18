import numpy as np

def findMax(matrix):
    """
    Нахождения максимамального по модулю элемента aij
    j > i
    """
    aij = [0] * n
    for i in range(n):
        for j in range(n):
            if j > i and np.abs(aij[0]) < np.abs(matrix[i][j]):
                aij = [matrix[i][j], i, j]
    return aij


def phi(matrix, a, i, j):
    """
    Функция нахождения sin и cos угла поворота
    a - max
    """
    phi = 0.5 * np.arctan(2 * a / (matrix[i][i] - matrix[j][j]))
    return phi


# Метод вращения Якоби
def jacoby(matrix, debug = False):
    _max = findMax(matrix)[0]
    x = np.eye(n)
    k = 0
    while np.abs(_max) > eps:
        print("Итерация", k)
        _max, i, j = findMax(matrix)
        print('max|aij| = ', _max)
        ph = phi(matrix, _max, i, j)

        sin, cos = np.sin(ph), np.cos(ph)
        print(f" Sin и cos угла поворота: sin(phi_{k})={sin}", f"cos(phi_{k})={cos}")
        U = np.eye(n)
        U[i][i] = U[j][j] = cos
        U[i][j] = -sin
        U[j][i] = sin
        print('U =', U)
        U_T = U.transpose()
        print('Transpose of U =', U_T)
        x = np.matmul(x, U)
        matrix = np.matmul(np.matmul(U_T, matrix), U)
        print(f"A({k+1})=U({k})^T*A({k})*U({k}) = ", matrix)
        k += 1
    for j in range(n):
        x[:, j] /= x[j][j]
    return matrix.diagonal(), x

eps = 0.01
A = np.array([[-8.0, 5.0, -7.0],
              [5.0, 1.0, 4.0],
              [-7.0, 4.0, 4.0]])
n = len(A)

print('Исходная матрица', A)
M, X = jacoby(A)
print('Собственные значения:')
for i in range(n):
    print(f'lamda_{i + 1} = ', M[i])
print('Найденные собственные вектора:')
for j in range(n):
    print(f"X{j + 1} =", X[:, j])
print('Проверка:')
print(np.linalg.eigh(A)[0])
vector = np.linalg.eigh(A)[1]/np.linalg.eigh(A)[1].diagonal()
for j in range(n):
    print(f'X{j + 1} = ', vector[:, j])
