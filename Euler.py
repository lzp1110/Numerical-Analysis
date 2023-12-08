import numpy as np

def func(x,y):
    return x*x + x - y

def CorrectFunc(x):
    return -np.exp(-x) + x*x - x + 1

def EulerPlus(x_n,y_n,h):
    _y = y_n + h*func(x_n,y_n)
    return y_n + h/2*(func(x_n,y_n) + func(x_n+h,_y))


def main():
    h = 0.1
    x = 0
    y = 0
    for i in range(5):
        x_n = x + i * h
        y = EulerPlus(x_n,y,h)
        print(y)
    print("y(0.5) = ",CorrectFunc(0.5))

if __name__ == "__main__":
    main()
