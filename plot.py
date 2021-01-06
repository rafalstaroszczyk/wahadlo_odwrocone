import numpy as np
import matplotlib.pyplot as plt


def plot():  # wyswietlanie wykresow
    number_of_plots = len(data_to_plot)  # ilosc podwykresow
    dict = {"i": 0, "t": 1, "phi": 2, "x": 3, "omega": 4, "v": 5, "epsilon": 6, "alfa": 7, "z": 8}  # kolumna w dane
    for i in range(number_of_plots):
        plt.subplot(number_of_plots, 1, i + 1)
        # wyswietla dane
        plt.plot(plot_data[:, 1], plot_data[:, dict[data_to_plot[i]]], color='g', lw=1, ls='-', label=data_to_plot[i])
        # wyswietla max i min
        localmax0 = local_max(plot_data[:, 1], plot_data[:, dict[data_to_plot[i]]])[0]
        localmax1 = local_max(plot_data[:, 1], plot_data[:, dict[data_to_plot[i]]])[1]
        localmin0 = local_min(plot_data[:, 1], plot_data[:, dict[data_to_plot[i]]])[0]
        localmin1 = local_min(plot_data[:, 1], plot_data[:, dict[data_to_plot[i]]])[1]

        plt.plot(localmax0, localmax1, color='r', lw=1, ls='-.')
        plt.plot(localmin0, localmin1, color='r', lw=1, ls='-.')

        plt.plot(local_max(localmax0, localmax1)[0], local_max(localmax0, localmax1)[1], color='b', lw=1, ls=':')
        plt.plot(local_min(localmin0, localmin1)[0], local_min(localmin0, localmin1)[1], color='b', lw=1, ls=':')

        plt.legend(loc='lower left')

    plt.show()


def local_max(t, x):
    localmaxt = []
    localmaxx = []
    temp1 = 0
    for i in range(len(x) - 1):
        temp2 = x[i]  # przesuniecie "testu"
        temp3 = x[i + 1]  # przesuniecie "testu"
        if temp2 >= temp1 and temp2 >= temp3:  # jesli jest w max lokalnym dodaje do listy
            localmaxt.append(t[i])
            localmaxx.append(x[i])
            temp1 = x[i]  # przesuniecie "testu"
        else:
            temp1 = x[i]  # przesuniecie "testu"
    return localmaxt, localmaxx  # lista wartosci t i lista wartosci funkcji


def local_min(t, x):
    localmint = []
    localminx = []
    temp1 = 0
    for i in range(len(x) - 1):
        temp2 = x[i]  # przesuniecie "testu"
        temp3 = x[i + 1]  # przesuniecie "testu"
        if temp2 <= temp1 and temp2 <= temp3:  # jesli jest w min lokalnym dodaje do listy
            localmint.append(t[i])
            localminx.append(x[i])
            temp1 = x[i]  # przesuniecie "testu"
        else:
            temp1 = x[i]  # przesuniecie "testu"
    return localmint, localminx  # lista wartosci t i lista wartosci funkcji


def main():
    data = np.loadtxt("data")
    plot(data, ("phi", "omega", "epsilon", "x", "v", "alfa"))


if __name__ == "__main__":
    main()