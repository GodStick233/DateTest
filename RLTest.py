import pandas as pd
import matplotlib.pylab as plt

def darw(filename):
    lx = pd.read_csv("L"+filename+".csv")
    ly = lx.pop('Data')
    rx = pd.read_csv("R"+filename+".csv")
    ry = rx.pop('Data')

    plt.plot(lx,ly,label='L_dates')
    plt.plot(rx, ry, label='R_dates')
    plt.legend()
    plt.savefig("RL"+filename+".png")
    plt.show()

if __name__ == '__main__':
    filename = input('Name:')
    darw(filename)
