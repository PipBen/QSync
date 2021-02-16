import numpy as np
import matplotlib.pylab as plt

def fft_time_to_freq(time):
    dt = time[1] - time[0]
    N = time.shape[0]
    freq = 2.*np.pi*np.fft.fftfreq(N, dt)  #
    return np.append(freq[N//2:], freq[:N//2])

def fft(y_t, t, return_frequencies=False):
    n_times = len(t)
    ft = t[-1] * np.fft.fft(y_t, n_times)
    ft = np.append(ft[n_times // 2:], ft[:n_times // 2])
    if return_frequencies ==True:
        frequencies = fft_time_to_freq(t)
        return frequencies, ft
    else:
        return ft

times = np.linspace(0, 100000, 100000)
omega = 1

yt = np.sin(omega * times)
freqs, ft = fft(yt, times, True)


plt.figure()

plt.plot(freqs, np.real(ft))
plt.show()

for j in range(6):
    print(j)