import numpy as np


def main():
    xlocal = 1/(2*np.pi)
    ylocal = 0.01

    angle1 = np.arctan2(-ylocal, -xlocal)
    angle2 = np.arctan2(-ylocal, 1 - xlocal)
    beta = angle2 - angle1

    print(beta)


main()