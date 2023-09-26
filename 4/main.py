import numpy as np
from matplotlib import pyplot as plt

# ВАРИАНТ 2

# -2*z /x - y/x^4
func = lambda x, y, z: -2*z/x - y/np.power(x, 4)


def plot(x, xn, y1, y2,_y1, _y2, h_1, h_2):
    fig, axs = plt.subplots(2)
    fig.suptitle('Метод Рунге-Кутты 4го порядка для ОДУ 2го порядка')
    XNEW1 = np.linspace(x, xn, int((xn - x) / h_1 + 1))
    XNEW2 = np.linspace(x, xn, int((xn - x) / h_2 + 1))
    axs[0].plot(XNEW1, y1, label='y(x) при h = 0.1')
    axs[0].plot(XNEW2, y2, label='y(x) при h = 0.01')
    axs[1].plot(XNEW1, _y1, label="y'(x) при h = 0.1")
    axs[1].plot(XNEW2, _y2, label="y'(x) при h = 0.01")

    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('y(x)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title('y\'(x)')
    plt.show()


class KuttaModified:
    def __init__(self, x0, y0, z0, boundaries):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.boundaries = boundaries


    def __call__(self, h):
        Y = [self.y0]
        Z = [self.z0]
        x0 = self.x0
        nodes = np.arange(self.boundaries[0], self.boundaries[1], h)
        print(f'Значение дифференциального уравнения в точке {x0:.5f}: {self.y0:.5f}')
        for node in nodes:
            x0 = node
            z = Z[-1]
            y = Y[-1]

            K1 = h*z
            L1 = h*func(x0, y, z)
            K2 = h*(z+0.5*L1)
            L2 = h*func(x0 + 0.5 * h, y + 0.5 * K1, z + 0.5 * L1)
            K3 = h*(z + 0.5 * L2)
            L3 = h*func(x0 + 0.5 * h, y + 0.5 * K2, z + 0.5 * L2)
            K4 = h*(z + L3)
            L4 = h*func(x0 + h, y + K3, z + L3)
            y = y + (1.0 / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4)
            z = z + (1.0 / 6.0) * (L1 + 2.0 * L2 + 2.0 * L3 + L4)
            # x0 = x0 + h
            Z.append(z)
            Y.append(y)
            print(f'Значение дифференциального уравнения в точке {x0:.5f}: {y:.5f}')
        return (Y, Z)


if __name__ == '__main__':
    Kutta = KuttaModified(1.0, 1.0, 2.0, [1.0, 2.0])
    basic_h1 = Kutta(0.1)
    basic_h2 = Kutta(0.01)

    print("="*50)
    print('h = 0.1')
    print(basic_h1[1])
    print("="*50)
    print('h = 0.01')
    print(basic_h2[1])

    plot(1.0, 2.0, basic_h1[0], basic_h2[0], basic_h1[1], basic_h2[1], 0.1, 0.01)
    plt.title("Аппроксимации")
    plt.legend()
    plt.show()
