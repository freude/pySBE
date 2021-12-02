import numpy as np
import matplotlib.pyplot as plt
from sbe.constants import *


def main(kk, epss1, epss2, A, B, C, D):

    q1 = np.linspace(A, B, 170)
    q2 = np.linspace(C, D, 170)
    q2 = q2 + (q2[3] - q2[2]) * 0.5

    def f(q1, q2, eps1, eps2, kkk):

        a = eps1 * kkk ** 2
        b = eps2
        f = q1 * q2 / np.sqrt((a + b * (q1 - q2) ** 2) * (a + b * (q1 + q2) ** 2))

        return f

    X, Y = np.meshgrid(q1, q2)
    fff = f(X, Y, epss1, epss2, kk)

    # plt.imshow(ans)
    # plt.show()

    return np.trapz(np.trapz(fff, q1, axis=1), q2, axis=0)


def gen_pot(k, eps1, eps2, k_lim1, k_lim2, k_lim3, k_lim4):
    l_k = np.size(k)
    V = np.zeros((l_k, l_k))

    for j1 in range(l_k):
        for j2 in range(l_k):
            # angular integration
            q = np.abs(k[j1] - k[j2])
            V[j1][j2] = main(q, eps1, eps2, k_lim1, k_lim2, k_lim3, k_lim4)

    # V[0, 0] = 0.0
    epss = 1.0

    # for j1 in range(1, l_k):
    #     V1[j1, j1] = V1[j1, j1 - 1] * 5

    return V * epss * e ** 2 / (4.0 * np.pi * (2*np.pi) * eps0)


def gen_pot1(k, eps1, eps2, G):
    l_k = np.size(k)
    V = np.zeros((l_k, l_k))

    for j1 in range(l_k):
        for j2 in range(l_k):
            # angular integration
            q = np.abs(k[j1] - k[j2])
            V[j1][j2] = 1.0 / (eps1 * q**2 + eps2 * G**2)

    # V[0, 0] = 0.0
    epss = 1.0

    # for j1 in range(1, l_k):
    #     V1[j1, j1] = V1[j1, j1 - 1] * 5

    return V * epss * e ** 2 / (4.0 * np.pi * (2*np.pi) * eps0)


if __name__ == '__main__':

    k = np.linspace(0, 5e9, 170)
    ans = []
    ans1 = []
    ans2 = []
    ans3 = []
    ans4 = []
    ans5 = []
    ans6 = []
    ans7 = []

    shift = 1e9 * 0

    for item in k:
        # ans.append(main(item, 0+shift, 0.1e9+shift, 0+shift, 0.1e9+shift))
        # ans1.append(main(item, 0+shift, 0.5e9+shift, 0+shift, 0.5e9+shift))
        # ans2.append(main(item, 0+shift, 1.0e9+shift, 0+shift, 1.0e9+shift))
        # ans3.append(main(item, 0+shift, 1.5e9+shift, 0+shift, 1.5e9+shift))
        # ans4.append(main(item, 0+shift, 2.0e9+shift, 0+shift, 2.0e9+shift))
        # ans5.append(main(item, 0+shift, 2.5e9+shift, 0+shift, 2.5e9+shift))
        # ans6.append(main(item, 0+shift, 3.0e9+shift, 0+shift, 3.0e9+shift))
        # ans7.append(main(item, 0+shift, 3.5e9+shift, 0+shift, 3.5e9+shift))

        ans.append(main(item, 0, 0.1e9, 0.1e9, 0.2e9))
        ans1.append(main(item, 0, 0.5e9, 0.5e9, 1.0e9))
        ans2.append(main(item, 0, 1.0e9, 1.0e9, 2.0e9))
        ans3.append(main(item, 0, 1.5e9, 1.5e9, 3.0e9))
        ans4.append(main(item, 0, 2.0e9, 2.0e9, 4.0e9))
        ans5.append(main(item, 0, 2.5e9, 2.5e9, 5.0e9))
        ans6.append(main(item, 0, 3.0e9, 3.0e9, 6.0e9))
        ans7.append(main(item, 0, 3.5e9, 3.5e9, 7.0e9))

        # ans1.append(main(item, 0+1e9, 1e9+1e9, 0+1e9, 1e9+1e9))
        # ans2.append(main(item, 0, 1e9, 0+1e9, 1e9+1e9))

    plt.subplot(121)
    plt.plot(np.log(ans))
    plt.plot(np.log(ans1))
    plt.plot(np.log(ans2))
    plt.plot(np.log(ans3))
    plt.plot(np.log(ans4))
    plt.plot(np.log(ans5))
    plt.plot(np.log(ans6))
    plt.plot(np.log(ans7))

    plt.subplot(122)
    plt.plot(ans)
    plt.plot(ans1)
    plt.plot(ans2)
    plt.plot(ans3)
    plt.plot(ans4)
    plt.plot(ans5)
    plt.plot(ans6)
    plt.plot(ans7)
    plt.show()
