import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from src.util import sample, symb_from_msg, demodulate, pick_sf_b, generate_msg, compute_ser, compute_ber


def compute_curve(sir, n_symbol=10):
    """
    Generate one instance of SER/SIR and BER/SIR curves by randomly picking two random (SF, B) and demodulating message
    1 and varying the SIR
    :param X: values of SIR
    :param n_symbol: Length of message in number of symbols
    :return: SER and BER values
    """
    sf1, bw1, sf2, bw2 = pick_sf_b()
    symbols, msg1, msg2 = generate_msg(sf1, bw1, sf2, bw2, n_symbol)
    y_ser = y_ber = np.zeros(len(sir))

    for j in range(len(sir)):
        # Generate messages and demodulate message 1
        msg = msg1 + sir[j] * msg2[:len(msg1)]
        dem = sample(demodulate(msg, n_symbol, sf1, bw1), bw1)
        translated = symb_from_msg(dem, n_symbol, bw1, sf1)
        # Compute SER and BER
        y_ser[j] = compute_ser(symbols, translated)
        y_ber[j] = compute_ber(symbols, translated)

    return y_ser, y_ber


def main(args):
    """
    Runs N_iter iterations of compute_curve, averages, saves and plots the results
    """
    n_iters = args.n_iters
    n_points = args.n_points
    n_symbol = args.n_symbol
    # Points in dB for SIRs
    X_dB = np.linspace(-5, 10, n_points, dtype=float)
    sir = np.sqrt(10 ** (X_dB - 1))

    sers = np.zeros(len(sir))
    bers = np.zeros(len(sir))

    for _ in tqdm(range(n_iters)):
        a, b = compute_curve(sir, n_symbol)
        sers += a
        bers += b

    sers /= n_iters
    bers /= n_iters

    # Save and plot results
    np.save("save.npy", (X_dB, sers, bers))

    plt.yscale('log')
    plt.xlabel("SIR")
    plt.ylabel("Error rate")
    plt.grid(True, which="both")
    plt.plot(X_dB, sers, label='SER')
    plt.plot(X_dB, bers, label='BER')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--n_iters', help="Number of iterations for plot (default=50)", required=False, type=int)
    parser.set_defaults(n_iters=50)
    parser.add_argument('-p', '--n_points', help="Number of points per plot (default=20)", required=False, type=int)
    parser.set_defaults(n_points=20)
    parser.add_argument('-s', '--n_symbol', help="Number of symbols per message (default=10)", required=False, type=int)
    parser.set_defaults(n_symbol=10)
    args = parser.parse_args()
    main(args)
