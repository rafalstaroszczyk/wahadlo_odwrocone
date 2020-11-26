import numpy as np


def data_input():
    offset_flag, symani = 0, 0
    while (True):
        try:
            f_prob = float(input("Podaj czestotliwosc probkowania [Hz]: "))
            t = float(input("Podaj czas symulacji [s]: "))
            phi = float(input("Podaj kat poczatkowy [st.]: "))
            omega = float(input("Podaj poczatkowa predkosc katowa [st./s]: "))
            x = float(input("Podaj poczatkowe polozenie punktu obrotu [m]: "))
            v = float(input("Podaj poczatkowa predkosc punktu obrotu [m/s]: "))
            a = float(input("Podaj amplitude drgan [m]: "))
            l = float(input("Podaj dlugosc wahadla [m]: "))
            g = float(input("Podaj przyspieszenie ziemskie [m/s^2]: "))
            f = float(input("Podaj czestotliwosc drgan [Hz]: "))
            while (offset_flag != 'y' and offset_flag != 'n'):
                offset_flag = input("Czy zamienic phi na pi-phi? (y/n): ")
            while (symani != 'y' and symani != 'n'):
                symani = input("Czy pokazac animacje symulacji? (y/n): ")
            dt = 1 / f_prob
            break
        except ValueError:
            print("Blad danych, sprobuj ponownie")
    return dt, t, phi * np.pi / 180, omega * np.pi / 180, x, v, a, l, g, f, offset_flag, symani


def epsilon(phi, alfa, z):  # przyspieszenie pionowe
    global a, f, g, l, dt
    acc = np.array((alfa, 4 * np.pi ** 2 * f ** 2 * z - g))  # wektor przyspieszenia: w lewo, w gore
    return 3 / 2 * 1 / l * np.dot(acc, np.array((-np.cos(phi), np.sin(phi))))


def alfa():  # przyspieszenie poziome
    return 0


def sym_ani(index):  # animacja symulacji
    global sym_state
    if index % 10000 == 0:
        if sym_state == 0:
            print("Symulowanie.")
            sym_state = 1
        elif sym_state == 1:
            print("Symulowanie..")
            sym_state = 2
        elif sym_state == 2:
            print("Symulowanie...")
            sym_state = 0


def simulate():
    global dt, t, n, phi0, omega0, x0, v0, a, l, g, f, data, symani
    # ustawianie wartosci poczatkowych
    data[0, 2] = phi0
    data[0, 3] = x0
    data[0, 4] = omega0
    data[0, 5] = v0
    data[0, 6] = -3 / 2 * g / l * np.sin(phi0)
    data[0, 7] = alfa()
    for i in range(n - 1):
        data[i + 1, 0:6] = data[i, 0:6] + np.array(
            (1, dt, data[i, 4] * dt, data[i, 5] * dt, data[i, 6] * dt, data[i, 7]))
        data[i + 1, 6] = epsilon(data[i, 2], data[i, 7], data[i, 8])
        data[i + 1, 7] = alfa()
        data[i + 1, 8] = a * np.sin(2 * np.pi * f * data[i, 1])
        if symani == 'y':
            sym_ani(data[i, 0])


"""
dt = 0.0001  # skok probkowania
t = 10  # czas symulacji
n = int(t / dt)  # ilosc probek
phi0 = 0.99 * np.pi  # kat poczatkowy
omega0 = 0  # predkosc katowa poczatkowa
x0 = 0  # polozenie poczatkowe poziome pktu obrotu
v0 = 0  # predkosc poczatkowa pozioma pktu obrotu
a = 0.01  # amplituda drgan
l = 1  # dlugosc wahadla
g = 10  # przysp. grawitacyjne
f = 100  # czestotliwosc drgan
offset_flag = 'y'  # czy ma byc przesuniecie w phi
symani = 'y'  # czy ma byc animacja symulacji
"""

dt, t, phi0, omega0, x0, v0, a, l, g, f, offset_flag, symani = data_input()
n = int(t / dt)
sym_state = 0  # czesc sym_ani(index)

data = np.zeros((n, 9))
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
simulate()
if offset_flag == 'y':
    data[:, 2] = np.ones((1, n)) * np.pi - data[:, 2]  # kat od OZ w prawo
    header = '%8s%11s%11s%11s%11s%11s%11s%11s%11s' % ('i', 't', 'pi-phi', 'x', 'omega', 'v', 'epsilon', 'alfa', 'z')
else:
    header = '%8s%11s%11s%11s%11s%11s%11s%11s%11s' % ('i', 't', 'phi', 'x', 'omega', 'v', 'epsilon', 'alfa', 'z')

np.savetxt("data", data, fmt="%10.5f", header=header)
