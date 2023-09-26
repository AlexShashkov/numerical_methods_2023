import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ВАРИАНТ 2

func = lambda x, y: -y/(2*x)+np.power(x, 2)


def plot(a, b, c_a, c_b, method):
    plt.grid(True)
    plt.plot(a[0], a[1], color = c_a, label = f"{method}, h = 0.1")
    plt.plot(b[0], b[1], color = c_b, label = f"{method}, h = 0.01")


class EulerSolve:
    def __init__(self, x0, y0, boundaries):
        self.x0 = x0
        self.y0 = y0
        self.boundaries = boundaries


    def _Euler(self, h):
        X = [self.x0]
        Y = [self.y0]
        nodes = np.arange(self.boundaries[0], self.boundaries[1], h)
        for node in nodes:
            x = X[-1]
            y = Y[-1]
            y = y + h*func(x, y)
            X.append(x+h)
            Y.append(y)
        return (X, Y)


    def _EulerImp(self, h):
        X = [self.x0]
        Y = [self.y0]
        nodes = np.arange(self.boundaries[0], self.boundaries[1], h)
        for node in nodes:
            x = X[-1]
            y = Y[-1]
            y_next = y + h*func(x, y)
            y = y + (h/2)*(func(x, y)+func(x+h, y_next))
            X.append(x+h)
            Y.append(y)
        return (X, Y)


    def __call__(self, h, improved=True):
        if improved: return self._EulerImp(h)
        return self._Euler(h)


Euler = EulerSolve(1.0, 1.0, [1.0, 2.0])
basic_h1 = Euler(0.1, False)
basic_h2 = Euler(0.01, False)
improved_h1 = Euler(0.1)
improved_h2 = Euler(0.01)

print("="*50)
print('h = 0.1')
print("Basic Euler")
print(basic_h1[1])
print("Improved Euler")
print(improved_h1[1])
print("="*50)
print('h = 0.01')
print("Basic Euler")
print(basic_h2[1])
print("Improved Euler")
print(improved_h2[1])


plot(basic_h1, basic_h2, "red", "green", "Эйлер")
plot(improved_h1, improved_h2, "blue", "magenta", "Эйлер с пересчетом")
plt.title("Аппроксимации")
plt.legend()
plt.show()
