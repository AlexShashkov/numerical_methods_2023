import numpy as np
from matplotlib import pyplot as plt

# ВАРИАНТ 2

func = lambda x, y: y*(2*y*np.log(x)-1-1)/x
# func = lambda x, y: (np.power(x, 2)*np.log(x) - y) / x

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
        for node in nodes[:3]:
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


def Adams(X, Y, h, boundaries):
    nodes = np.arange(boundaries[0], boundaries[1], h)[3:]
    for node in nodes:
        y = Y[-1] + (h/24)*(55*func(X[-1], Y[-1])-59*func(X[-2], Y[-2])+37*func(X[-3], Y[-3])-9*func(X[-4], Y[-4]))
        Y.append(y)
        X.append(X[-1]+h)
    return (X, Y)




# Kutta = KuttaSolve(1.0, 0.3333333, [1.0, 2.0])
Kutta = KuttaSolve(1.0, 1.0, [1.0, 2.0])
basic_h1 = Kutta(0.1)
basic_h2 = Kutta(0.01)

adams_h1 = Adams(*basic_h1, 0.1, [1.0, 2.0])
adams_h2 = Adams(*basic_h2, 0.01, [1.0, 2.0])

print(adams_h1)
print(adams_h2)
print("eee")

plot(adams_h1, adams_h2, "red", "green", "Метод Адамса")
plt.title("Аппроксимации")
plt.legend()
plt.show()
