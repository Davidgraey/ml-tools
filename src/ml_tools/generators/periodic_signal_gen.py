import numpy as np

def make_phase_mix_dataset(n_samples=128, signal_len=128, f1=3, f2=11):
    t = np.linspace(0, 1, signal_len, endpoint=False)
    X = []
    Y = []

    for _ in range(n_samples):
        phase1 = np.random.rand() * 2 * np.pi
        phase2 = np.random.rand() * 2 * np.pi

        x = np.sin(2 * np.pi * f1 * t + phase1) + 0.5 * np.sin(
            2 * np.pi * f2 * t + phase2
        )

        y = 2.0 * np.sin(2 * np.pi * f1 * t + phase1) + 0.1 * np.sin(
            2 * np.pi * f2 * t + phase2
        )

        X.append(x)
        Y.append(y)

    X, Y = np.array(X), np.array(Y)
    X = X - np.mean(X)
    Y = Y - np.array(Y)

    return X, Y


def make_multifreq_dataset(
    batch_size=16,
    seq_len=64,
    hidden_dim=8,
    noise_std=0.1,
):
    t = np.linspace(0, 1, seq_len)

    low_freq = np.sin(2 * np.pi * 2 * t)  # low freq
    high_freq = np.sin(2 * np.pi * 10 * t)  # high freq

    X = np.zeros((batch_size, seq_len, hidden_dim))
    Y = np.zeros_like(X)

    for b in range(batch_size):
        choose_low = np.random.rand() < 0.5

        signal = low_freq if choose_low else high_freq
        signal = signal + noise_std * np.random.randn(seq_len)

        # embed signal in channel 0
        X[b, :, 0] = signal

        # query content encodes the choice
        X[b, :, 1] = 1.0 if choose_low else -1.0

        # target: clean signal only
        Y[b, :, 0] = low_freq if choose_low else high_freq

    return X, Y
