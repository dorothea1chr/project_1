import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby1, bilinear, freqz

# ------------------------------------------
# Δεδομένα άσκησης
# ------------------------------------------
wc = 2 * np.pi * 600       # αναλογική cutoff συχνότητα [rad/s]
Ts = 1/4000                # περίοδος δειγματοληψίας
fs = 1 / Ts                # συχνότητα δειγματοληψίας [Hz]
rp = 3.0                   # passband ripple [dB]
orders = [3, 7]            # τάξεις φίλτρων
# ------------------------------------------

plt.figure(figsize=(8, 5))

for N, style, color in zip(orders, ['-', '--'], ['C0', 'C1']):
    # Αναλογικό Chebyshev Type I Highpass φίλτρο
    b_a, a_a = cheby1(N, rp, wc, btype='high', analog=True)

    # Μετασχηματισμός bilinear σε ψηφιακό φίλτρο
    b_z, a_z = bilinear(b_a, a_a, fs=fs)

    # Απόκριση συχνότητας (256 σημεία)
    w, h = freqz(b_z, a_z, worN=256)

    # Κανονικοποίηση διαστήματος [0, 1]
    w_norm = w / np.pi

    # Μετατροπή σε dB
    H_dB = 20 * np.log10(np.abs(h) + 1e-12)

    # Σχεδίαση
    plt.plot(w_norm, H_dB, linestyle=style, color=color,
             label=f'Chebyshev I Highpass N={N}')

plt.xlabel('Κανονικοποιημένη συχνότητα ω/π (0..1)')
plt.ylabel('Πλάτος (dB)')
plt.title('Αποκρίσεις συχνότητας Chebyshev Highpass (rp=3dB)')
plt.grid(True)
plt.legend()
plt.ylim(-80, 5)
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
