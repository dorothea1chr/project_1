import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, tf2zpk

# Συντελεστές αριθμητή και παρονομαστή H(z)
b = [0.6, 0.06, -0.12]
a = [1, -0.25, -0.125]

# Ζεύγος πόλων και μηδενικών
zeros, poles, _ = tf2zpk(b, a)

# Συνάρτηση σχεδίασης zplane
def zplane(zeros, poles):
    plt.figure()
    plt.title('Pole-Zero Plot')
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', facecolors='none', edgecolors='b', label='Zeros')
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Poles')
    unit_circle = plt.Circle((0,0), 1, color='black', fill=False, ls='dashed')
    plt.gca().add_artist(unit_circle)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

zplane(zeros, poles)

# Δημιουργία διανύσματος συχνοτήτων στο διάστημα [-pi, pi] με βήμα pi/128
w = np.arange(-np.pi, np.pi + np.pi/128, np.pi/128)

# Απόκριση συχνότητας με freqz
w_freqz, h = freqz(b, a, worN=w)

# Απεικόνιση μεγέθους και φάσης
plt.figure()
plt.subplot(2,1,1)
plt.plot(w_freqz, 20 * np.log10(abs(h)))
plt.title('Magnitude Response')
plt.xlabel('Frequency (radians)')
plt.ylabel('Magnitude (dB)')
plt.grid()

plt.subplot(2,1,2)
plt.plot(w_freqz, np.angle(h))
plt.title('Phase Response')
plt.xlabel('Frequency (radians)')
plt.ylabel('Phase (radians)')
plt.grid()

plt.tight_layout()
plt.show()

# Προσθήκη πόλου στο z=1 σημαίνει πολλαπλασιασμός του παρονομαστή με (1 - z^-1)
a_new = np.convolve(a, [1, -1])  # Νέος παρονομαστής με επιπλέον πόλο
b_new = b  # Ο αριθμητής παραμένει ίδιος

# Απόκριση συχνότητας με νέο σύστημα
h = freqz(b_new, a_new, worN=w)

plt.figure()
plt.subplot(2,1,1)
plt.plot(w_freqz, 20 * np.log10(abs(h)))
plt.title('Magnitude Response with Extra Pole at z=1')
plt.xlabel('Frequency (radians)')
plt.ylabel('Magnitude (dB)')
plt.grid()

plt.subplot(2,1,2)
plt.plot(w_freqz, np.angle(h))
plt.title('Phase Response with Extra Pole at z=1')
plt.xlabel('Frequency (radians)')
plt.ylabel('Phase (radians)')
plt.grid()

plt.tight_layout()
plt.show()