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
            while (offset_flag != 'y' and offset_flag != 'n'):  # wymuszenie y lub n
                offset_flag = input("Czy zamienic phi na pi-phi? (y/n): ")
            while (symani != 'y' and symani != 'n'):  # wymuszenie y lub n
                symani = input("Czy pokazac animacje symulacji? (y/n): ")
            break
        except ValueError:
            print("Blad danych, sprobuj ponownie")
    return f_prob, t, phi * np.pi / 180, omega * np.pi / 180, x, v, a, l, g, f, offset_flag, symani


def epsilon(phi, alfa, z):  # przyspieszenie pionowe
    global a, f, g, l, dt
    acc = np.array((alfa, 4 * np.pi ** 2 * f ** 2 * z - g))  # wektor przyspieszenia: w lewo, w gore
    return 3 / 2 * 1 / l * np.dot(acc, np.array((-np.cos(phi), np.sin(phi))))


def alfa(phi, x, omega, z):  # przyspieszenie poziome
    global f, g, l, alfa_prop
    try:
        return alfa_prop * ((4 * np.pi ** 2 * f ** 2 * z - g) * np.sin(phi) - (l * omega ** 2) / (3 * phi - 3 * np.pi)) / np.cos(phi)
    except ZeroDivisionError:
        return 0
    # return -alfa_prop * g * np.tan(phi)


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
    global f_prob, t, n, phi0, omega0, x0, v0, a, l, g, f, data, symani
    # ustawianie wartosci poczatkowych
    data[0, 2] = phi0
    data[0, 3] = x0
    data[0, 4] = omega0
    data[0, 5] = v0
    data[0, 6] = -3 / 2 * g / l * np.sin(phi0)
    data[0, 7] = alfa(data[0, 2], data[0, 3], data[0, 4], data[0, 8])
    for i in range(n - 1):
        data[i + 1, 0:6] = data[i, 0:6] + np.array(
            (1, 1 / f_prob, data[i, 4] / f_prob, data[i, 5] / f_prob, data[i, 6] / f_prob, data[i, 7] / f_prob))
        data[i + 1, 6] = epsilon(data[i, 2], data[i, 7], data[i, 8])
        data[i + 1, 7] = alfa(data[i, 2], data[i, 3], data[i, 4], data[i, 8])
        data[i + 1, 8] = a * np.sin(2 * np.pi * f * data[i, 1])
        """
        i += 1
        t += dt
        phi += omega * dt
        x += v * dt
        omega += epsilon * dt
        v += alfa * dt
        epsilon = epsilon(phi, alfa, z)
        alfa = alfa(phi)
        z = a * sin(2 * pi * f * t)
        """
        if symani == 'y':
            sym_ani(data[i, 0])


"""
f_prob = 10000  # skok probkowania
t = 10  # czas symulacji
n = int(t * f_prob)  # ilosc probek
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

f_prob, t, phi0, omega0, x0, v0, a, l, g, f, offset_flag, symani = data_input()
n = int(t * f_prob)

# te zmienne musza byc poza funkcja bo funkcje sa wywolywane wielokrotnie i wynik jest inny
sym_state = 0  # czesc sym_ani(index)
alfa_prop = 10  # czesc alfa()

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
if offset_flag == 'y':  # inne oznaczenia w pliku: phi i pi - pih
    data[:, 2] = np.ones((1, n)) * np.pi - data[:, 2]  # kat od OZ w prawo
    header = '%8s%11s%11s%11s%11s%11s%11s%11s%11s' % ('i', 't', 'pi-phi', 'x', 'omega', 'v', 'epsilon', 'alfa', 'z')
else:
    header = '%8s%11s%11s%11s%11s%11s%11s%11s%11s' % ('i', 't', 'phi', 'x', 'omega', 'v', 'epsilon', 'alfa', 'z')

np.savetxt("data", data, fmt="%10.5f", header=header)  # zapisywanie do pliku
