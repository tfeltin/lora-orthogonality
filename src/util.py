import numpy as np

# ----- CSS PROPERTIES -----

fc = 250e3  # CARRIER FREQUENCY
ech = fc * (2 ** 5)  # SAMPLING RATE

BANDWIDTHS = [125e3, 250e3, 500e3]
SFACTORS = [7, 8, 9, 10, 11, 12]


# --- MESSAGE GENERATION ---

def is_orthogonal(sf1, bw1, sf2, bw2):
    """
    Checks if two (SF, B) couples are orthogonal by comparing the slopes in spectrogram
    :param sf1, sf2: Spreading Factor
    :param bw1, bw2: Badwidth
    :return: Boolean, whether the couple is orthogonal or not
    """
    p1 = (bw1 * bw1) / (2 ** sf1)
    p2 = (bw2 * bw2) / (2 ** sf2)
    return p1 != p2


def pick_sf_b():
    """
    Generates orthogonal pair of (SF, B)
    :return: Two orthogonal (SF, B) pairs
    """
    while True:
        sf1 = np.random.choice(SFACTORS, 1)[0]
        sf2 = np.random.choice(SFACTORS, 1)[0]
        bw1 = np.random.choice(BANDWIDTHS, 1)[0]
        bw2 = np.random.choice(BANDWIDTHS, 1)[0]
        if is_orthogonal(sf1, bw1, sf2, bw2):
            break
    return sf1, bw1, sf2, bw2


def prof_freq(B, SF, k, mu):
    """
    Returns the profile of the frequency of a single encoded symbol
    :param B: Bandwith
    :param SF: Spreading Factor
    :param k: Value of symbol, between 0 and 2**SF
    :param mu: if -1 returns downchirp
    :return: f, frequence profile
    """
    N = 2 ** SF
    Ts = N / B
    f0 = fc - (B / 2)
    f1 = fc + (B / 2)
    t = np.arange(0, Ts, 1 / ech)
    lk = k / B
    a = mu * B / Ts
    f = np.zeros(len(t))
    if mu == -1:
        f0, f1 = f1, f0
    for i in range(len(t)):
        ti = t[i]
        if ti < lk:
            f[i] = a * (ti - lk) + f1
        else:
            f[i] = a * (ti - lk) + f0
    return f


def phase_freq(f):
    """
    Returns the phase of the signal
    :param f: Frequency over time
    :return: Phase over time
    """
    n = len(f)
    phi = np.zeros(n)
    for i in range(1, n):
        phii = phi[i - 1] + 2 * np.pi * f[i - 1] * (1 / ech)
        phi[i] = phii % (2 * np.pi)
        if phi[i] > np.pi:
            phi[i] = phi[i] - 2 * np.pi
    return phi


def sig_phase(phi):
    """
    Generate signal in amplitude from phase
    :param phi: Phase over time
    :return: Amplitude of signal over time
    """
    n = len(phi)
    s = np.zeros(n, dtype=complex)
    for i in range(n):
        s[i] = np.exp(1j * phi[i])
    return s


def msg_chirp(length, sf, bw):
    """
    Creates a random message and returns its temporal representation
    :param length: Number of symbols in generated message
    :param sf: Spreading Factor
    :param bw: Bandwidth
    :return: Amplitude of signal over time
    """
    N = 2 ** sf
    symbol = np.random.randint(0, N, length)
    signal = np.array([])
    phase = np.array([])
    for i in range(len(symbol)):  # tqdm
        phasee = phase_freq(prof_freq(bw, sf, symbol[i], 1))
        nc = sig_phase(phasee)
        signal = np.concatenate([signal, nc])
        phase = np.concatenate([phase, phasee])
    return symbol, signal, phase


def generate_msg(sf1, bw1, sf2, bw2, n_symbols):
    """
    Generate two signals in amplitude over time
    :param sf1, sf2: Spreading factors
    :param bw1, bw2: Bandwidths
    :param n_symbols: Number of symbols in message 1 (Signal to demodulate)
    :return: List of symbols in message 1 and the two messages in amplitude over time
    """
    t1 = (2 ** sf1) / bw1
    t2 = (2 ** sf2) / bw2
    n2_symbols = n_symbols * (int(t1 / t2) + 1)
    symbols, msg1, _ = msg_chirp(n_symbols, sf1, bw1)
    _, msg2, _ = msg_chirp(n2_symbols, sf2, bw2)

    return symbols, msg1, msg2


# ------- DEMODULATION -------


def down_chirp(length, sf, bw):
    """
    Creates down chirp of corresponding length for demodulation
    :param length: Length of message
    :param sf: Spreading Factor
    :param bw: Bandwidth
    :return: Down chirp in amplitude over time
    """
    nc = sig_phase(phase_freq(prof_freq(bw, sf, 0, -1)))
    out = np.concatenate([nc] * length)
    return out


def demodulate(msg, length, sf, b):
    """
    Creates corresponding downchirp and multiply with signal to demodulate
    :param msg: the signal to demodulate
    :param length: number of symbols in message
    :param sf: Spreading factor
    :param b: Bandwidth
    :return: Multiplication in amplitude over time
    """
    down = down_chirp(length, sf, b)
    return msg * down


def sample(signal, b):
    """
    Sample signal at specified frequence
    :param signal: Signal to sample
    :param b: Sampling frequency
    :return: Sampled signal
    """
    sample_r = np.arange(0, len(signal), step=int(ech / b))
    return signal[sample_r]


def freq_from_fft(sig, fs):
    """
    Fourier analysis on symbol
    :param sig: Signal with single symbol
    :param fs: Sampling frequency
    :return: Measured frequence
    """
    f = np.fft.fft(sig)
    i = np.argmax(abs(f))
    return fs * i / len(sig)


def symb_from_msg(msg, length, b, sf):
    """
    Return symbol values of demodulation
    :param msg: Signal to demodulate
    :param length: Number of symbols
    :param b: Bandwidth
    :param sf: Spreading Factor
    :return: Array of integers between 0 and 2**sf-1 corresponding to symbols
    """
    frequencies = []
    step = int(len(msg) / length)
    symbols = []
    Ts = (2 ** sf) / b
    for i in range(length):
        s_msg = msg[i * step:(i + 1) * step]
        frequencies += [freq_from_fft(s_msg, b)]
        symbols += [2 ** sf - int(round(Ts * frequencies[i]))]
    return symbols


# -------- OTHER ---------

def bit_dif(n1, n2):
    """
    Compute difference between values in number of bits
    :param n1, n2: Values to compare as integers
    :return: Number of differing bits
    """
    return len("{0:b}".format(n1 ^ n2).replace('0', ''))


def compute_ser(symbols, translated):
    ser = 0
    for i in range(len(translated)):
        if translated[i] != symbols[i]:
            ser += 1
    return (len(translated) - ser) / len(translated)


def compute_ber(symbols, translated):
    ber = 0
    for i in range(len(translated)):
        if translated[i] != symbols[i]:
            ber += bit_dif(translated[i], symbols[i])
    return (len(translated) - ber) / len(translated)
