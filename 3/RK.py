import numpy as np
from matplotlib import pyplot as plt

# ВАРИАНТ 2

func = lambda x, y: (-y*x+x*(x**2.0+1))/(x**2.0+1)


def plot(a, b, c_a, c_b, method):
    plt.grid(True)
    plt.plot(a[0], a[1], color = c_a, label = f"{method}, h = 0.1")
    plt.plot(b[0], b[1], color = c_b, label = f"{method}, h = 0.01")


class KuttaSolve:
    def __init__(self, x0, y0, boundaries):
        self.x0 = x0
        self.y0 = y0
        self.boundaries = boundaries


    def __call__(self, h, improved=True):
        X = [self.x0]
        Y = [self.y0]
        nodes = np.arange(self.boundaries[0], self.boundaries[1], h)
        for node in nodes:
            x = X[-1]
            y = Y[-1]

            K1 = h*func(x, y)
            K2 = h*func(x+h/2, y+K1/2)
            K3 = h*func(x+h/2, y+K2/2)
            K4 = h*func(x+h, y+K3)
            D_y = (K1+2*K2+2*K3+K4)/6
            y = y + D_y
            X.append(x+h)
            Y.append(y)
        return (X, Y)


if __name__ == '__main__':
    Kutta = KuttaSolve(0.0, 1.0, [1.0, 2.0])
    basic_h1 = Kutta(0.1)
    basic_h2 = Kutta(0.01)


    plot(basic_h1, basic_h2, "red", "green", "Рунге-Кутты")
    plt.title("Аппроксимации")
    plt.legend()
    plt.show()
