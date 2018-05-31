import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from util import sample, symb_from_msg, demodulate, pick_sf_b, pick_no_sf_b, generate_msg, compute_ser, compute_ber


def compute_curve(sir, n_symbol=10, non_orthogonal=False):
    """
    Generate one instance of SER/SIR and BER/SIR curves by randomly picking two random (SF, B) and demodulating message
    1 and varying the SIR
    :param X: values of SIR
    :param n_symbol: Length of message in number of symbols
    :return: SER and BER values
    """
    if non_orthogonal:
        sf1, bw1, sf2, bw2 = pick_no_sf_b()
    else:
        sf1, bw1, sf2, bw2 = pick_sf_b()
    symbols, msg1, msg2 = generate_msg(sf1, bw1, sf2, bw2, n_symbol)
    y_ser = np.zeros(len(sir))
    y_ber = np.zeros(len(sir))

    for j in range(len(sir)):
        # Generate messages and demodulate message 1
        msg = msg1 + sir[j] * msg2[:len(msg1)]
        dem = sample(demodulate(msg, n_symbol, sf1, bw1), bw1)
        translated = symb_from_msg(dem, n_symbol, bw1, sf1)
        # Compute SER and BER
        y_ser[j] = compute_ser(symbols, translated)
        y_ber[j] = compute_ber(symbols, translated, sf1)

    return y_ser, y_ber


def main(args):
    """
    Runs N_iter iterations of compute_curve, averages, saves and plots the results
    """
    n_iters = args.n_iters
    n_points = args.n_points
    n_symbol = args.n_symbol
    non_orthogonal = args.non_orthogonal
    save_file = "save.npy"

    print()
    if non_orthogonal:
        print("Generating %s pairs of non orthogonal signals:" % n_iters)
    else:
        print("Generating %s pairs of orthogonal signals:" % n_iters)
    print("\t- %s points per curve" % n_points)
    print("\t- %s symbols per message" % n_symbol)
    print()
    print("Running...")

    # Points in dB for SIRs
    X_dB = np.linspace(-3, 7, n_points, dtype=float)
    sir = np.sqrt(10 ** (X_dB - 1))

    sers = np.zeros(len(sir))
    bers = np.zeros(len(sir))

    for _ in tqdm(range(n_iters)):
        a, b = compute_curve(sir, n_symbol=n_symbol, non_orthogonal=non_orthogonal)
        sers += a
        bers += b

    sers /= n_iters
    bers /= n_iters

    # Save and plot results
    np.save(save_file, (X_dB, sers, bers))
    print("Saved results in %s" % save_file)

    plt.yscale('log')
    plt.xlabel("Signal Interference Rate")
    plt.ylabel("Symbol Error Rate")
    plt.grid(True, which="both")
    plt.plot(X_dB, sers, label='SER', color='black', lw=3)
    plt.show()

    plt.yscale('log')
    plt.xlabel("Signal Interference Rate")
    plt.ylabel("Bit Error Rate")
    plt.grid(True, which="both")
    plt.plot(X_dB, bers, label='BER', color='black', lw=3)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--n_iters', help="Number of iterations (default=50)", required=False, type=int)
    parser.set_defaults(n_iters=20)
    parser.add_argument('-p', '--n_points', help="Number of points per plot (default=50)", required=False, type=int)
    parser.set_defaults(n_points=50)
    parser.add_argument('-s', '--n_symbol', help="Number of symbols per message (default=10)", required=False, type=int)
    parser.set_defaults(n_symbol=10)
    parser.add_argument('-no', '--non_orthogonal', help="Set flag to switch to non orthogonal signals", required=False,
                        action='store_true')
    parser.set_defaults(non_orthogonal=False)
    args = parser.parse_args()
    main(args)
