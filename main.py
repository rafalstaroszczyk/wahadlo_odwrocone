import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def data_input():  # wejscie danych
    f_prob, t, a, l, g, f, offset_flag, symani, hor_mov, f_anim, wsp_tar, m = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # deklaracja zmiennych
    while (True):
        try:
            while f_prob <= 0:
                f_prob = float(input("Podaj czestotliwosc probkowania [Hz]: "))
            while t <= 0:
                t = float(input("Podaj czas symulacji [s]: "))
            phi = float(input("Podaj kat poczatkowy [st.]: "))
            omega = float(input("Podaj poczatkowa predkosc katowa [st./s]: "))
            x = float(input("Podaj poczatkowe polozenie punktu obrotu [m]: "))
            v = float(input("Podaj poczatkowa predkosc punktu obrotu [m/s]: "))
            while a < 0:
                a = float(input("Podaj amplitude drgan [m]: "))
            while l <= 0:
                l = float(input("Podaj dlugosc wahadla [m]: "))
            while g <= 0:
                g = float(input("Podaj przyspieszenie ziemskie [m/s^2]: "))
            while f <= 0:
                f = float(input("Podaj czestotliwosc drgan [Hz]: "))
            while (offset_flag != 'y' and offset_flag != 'n'):  # wymuszenie y lub n
                offset_flag = input("Czy zamienic phi na pi-phi? (y/n): ")
            while (symani != 'y' and symani != 'n'):  # wymuszenie y lub n
                symani = input("Czy pokazac animacje symulacji? (y/n): ")
            while (hor_mov != 'y' and hor_mov != 'n'):  # ruch poziomy
                hor_mov = input("Czy pozwolic na ruch poziomy? (y/n): ")
            all_data_to_plot = ["phi", "omega", "epsilon", "x", "v", "alfa"]  # lista dozwolonych danych
            data_to_plot = input("Wypisz dane do wyswietlenia (phi/omega/epsilon/x/v/alfa): ")
            data_to_plot = data_to_plot.replace(',', ' ').replace('.', ' ').replace('/', ' ').replace(';', ' ').replace(
                ':', ' ').split(' ')  # zamiana ',', '.', '/', ';', ':' na ' '
            data_to_plot = list(filter(None, data_to_plot))  # usuwanie pustych elementow listy
            data_to_plot = list(dict.fromkeys(data_to_plot))  # usuwanie duplikatow
            data_to_plot = [item for item in data_to_plot if item in all_data_to_plot]  # filtrowanie danych
            vt = float(input("Podaj predkosc wyprostowania [m/s]: "))
            while f_anim <= 0:
                f_anim = float(input("Podaj czestotliwosc animacji: "))
            while wsp_tar < 0:
                wsp_tar = float(input("Podaj wspolczynnik tarcia: "))
            while m < 0:
                m = float(input("Podaj mase wahadla [kg]: "))
            break
        except ValueError:
            print("Blad danych, sprobuj ponownie")
    return f_prob, t, phi * np.pi / 180, omega * np.pi / 180, x, v, a, l, g, f, offset_flag, symani, hor_mov, data_to_plot, vt, f_anim


def epsilon(phi, omega, alfa, z):  # przyspieszenie katowe
    acc = np.array((alfa, 4 * np.pi ** 2 * f ** 2 * z - g))  # wektor przyspieszenia: w lewo, w gore
    return 3 / 2 * 1 / l * np.dot(acc, np.array((-np.cos(phi), np.sin(phi)))) - 3 * wsp_tar * omega / 4 / m


def alfa(phi, omega, z):  # przyspieszenie poziome
    # epsilon = kphi * (phit - phi) + komega * (0 - omega)
    kphi = 3
    komega = 4  # dla komega ** 2 < 4 * kphi wystapia oscylacje
    if hor_mov == 'y':
        try:
            return ((4 * np.pi ** 2 * f ** 2 * z - g) * np.sin(phi) + 2 * l / 3 * (
                    kphi * (phi - phit) + komega * omega)) / np.cos(phi)
        except ZeroDivisionError:
            return 0
    else:
        return 0


def sym_ani(index):  # pokazanie % symulacji
    global sym_state
    if 100 * index % n == 0:
        print("%3i%%" % sym_state)
        sym_state += 1


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


def init():
    ax.set_xlim(-1.5 * l, 1.5 * l)
    ax.set_ylim(-1.5 * l, 1.5 * l)
    return line,


def animate_frames(i):
    line.set_data([x1[i], x2[i]], [y1[i], y2[i]])
    return line,


def animate():
    global x1, x2, y1, y2, line, ax, anim
    frame_data = data[0:int(n):int(f_prob/f_anim), 0:9]
    np.savetxt("frame_data", frame_data, fmt="%10.5f")
    x1 = frame_data[:, 3]
    y1 = frame_data[:, 8]
    x2 = x1 + l * np.sin(frame_data[:, 2])
    y2 = y1 - l * np.cos(frame_data[:, 2])

    fig, ax = plt.subplots()
    line, = plt.plot([], [], animated=True)

    anim = ani.FuncAnimation(fig, animate_frames, frames=range(int(n * f_anim / f_prob)), init_func=init, blit=True, interval=int(1000/f_anim))


def simulate():
    global phit, alfa_mode
    # ustawianie wartosci poczatkowych
    data[0, 2] = phi0
    data[0, 3] = x0
    data[0, 4] = omega0
    data[0, 5] = v0
    data[0, 8] = 0
    data[0, 7] = alfa(data[0, 2], data[0, 4], data[0, 8])
    data[0, 6] = epsilon(data[0, 2], data[0, 4], data[0, 7], data[0, 8])
    for i in range(n - 1):
        data[i + 1, 0:2] = data[i, 0:2] + np.array((1, 1 / f_prob))
        data[i + 1, 8] = a * np.sin(2 * np.pi * f * data[i + 1, 1])
        data[i + 1, 7] = alfa(data[i, 2], data[i, 4], data[i, 8])
        data[i + 1, 6] = epsilon(data[i, 2], data[i, 4], data[i, 7], data[i, 8])
        data[i + 1, 4:6] = data[i, 4:6] + np.array(((data[i, 6] + data[i + 1, 6]) / (2 * f_prob), (data[i, 7] + data[i + 1, 7]) / (2 * f_prob)))
        data[i + 1, 2:4] = data[i, 2:4] + np.array(((data[i, 4] + data[i + 1, 4]) / (2 * f_prob), (data[i, 5] + data[i + 1, 5]) / (2 * f_prob)))
        """
        i += 1
        t += dt
        z = a * sin(2 * pi * f * t)
        alfa = alfa(phi)
        epsilon = epsilon(phi, alfa, z)
        omega[i+1] += ( epsilon[i] + epsilon[i + 1]) / 2 * dt
        v[i + 1] += (alfa[i] + alfa[i + 1]) / 2 * dt
        phi[i + 1] += (omega[i] + omega[i + 1]) / 2 * dt
        x[i + 1] += (v[i] + v[i + 1]) / 2 * dt
        """
        if hor_mov == 'y' and data[i + 1, 5] >= vt:
            phit = np.pi
        if symani == 'y':
            sym_ani(data[i, 0])


"""
f_prob = 100000  # skok probkowania
t = 10  # czas symulacji
n = int(t * f_prob)  # ilosc probek
phi0 = 150 / 180 * np.pi  # kat poczatkowy
omega0 = 0  # predkosc katowa poczatkowa
x0 = 0  # polozenie poczatkowe poziome pktu obrotu
v0 = 0  # predkosc poczatkowa pozioma pktu obrotu
a = 0.03  # amplituda drgan
l = 1  # dlugosc wahadla
g = 10  # przysp. grawitacyjne
f = 50  # czestotliwosc drgan
offset_flag = 'y'  # czy ma byc przesuniecie w phi
symani = 'y'  # czy ma byc animacja symulacji
hor_mov = 'n'  # czy ma byc ruch poziomy
data_to_plot = ["phi", "omega", "epsilon", "x", "v", "alfa"]
vt = 10  # predkosc po ktorej nastepuje ustawienie pionowe
f_anim = 60
wsp_tar = 0.1
m = 0.2
"""
f_prob, t, phi0, omega0, x0, v0, a, l, g, f, offset_flag, symani, hor_mov, data_to_plot, vt, f_anim, wsp_tar, m = data_input()
n = int(t * f_prob)  #liczba probek


# te zmienne musza byc poza funkcja bo funkcje sa wywolywane wielokrotnie i wynik jest inny
sym_state = 0  # czesc sym_ani(index)
phit = phi0  # kat docelowy

data = np.zeros((n, 9))
plot_data = np.zeros((n, 9))

"""
data[:, 0] = i - index
data[:, 1] = t - czas
data[:, 2] = phi - kat od -OZ w lewo
data[:, 3] = x - pol. poziome pktu obrotu
data[:, 4] = omega - pred. katowa
data[:, 5] = v - predkosc pozioma pktu obrotu
data[:, 6] = epsilon - przysp. katowe
data[:, 7] = alfa - przysp. poziome pkt. obrotu
data[:, 8] = z - pol. pionowe pktu obrotu
"""
if hor_mov == 'y':
    wsp_tar = 0

simulate()
print("%3i%%" % sym_state)
plot_data[:, :] = data[:, :]

if offset_flag == 'y':  # inne oznaczenia w pliku: phi i pi - phi
    plot_data[:, 2] = np.ones((1, n)) * np.pi - data[:, 2]  # kat od OZ w prawo
    header = '%8s%11s%11s%11s%11s%11s%11s%11s%11s' % ('i', 't', 'pi-phi', 'x', 'omega', 'v', 'epsilon', 'alfa', 'z')
else:
    header = '%8s%11s%11s%11s%11s%11s%11s%11s%11s' % ('i', 't', 'phi', 'x', 'omega', 'v', 'epsilon', 'alfa', 'z')

np.savetxt("data", plot_data, fmt="%10.5f", header=header)  # zapisywanie do pliku
plot()
animate()
plt.show()
