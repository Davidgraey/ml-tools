from numpy.typing import NDArray
import numpy as np
from ml_tools.models.layers.layers import (
    Layer,
    FullyConnectedLayer,
    FourierLayer,
    NormalizeLayer,
)


class FourierAttention(Layer):
    def __init__(self, ni: int, no: int, use_2d: bool = True):
        super().__init__()
        self.fftlayer = FourierLayer(use_2d)
        self.norm_a = NormalizeLayer(ni=ni, shift_scale=False)
        self.fc = FullyConnectedLayer(ni=ni, no=no, activation_type="relu")
        self.norm_b = NormalizeLayer(ni=no, shift_scale=True)

        self.hidden_size: int = 1

    def forward(self, x_data: NDArray):
        self.hidden_size = x_data.shape[-1]

        fft_x = self.norm_a(self.fftlayer(x_data) + x_data)

        self.output = self.norm_b(self.fc(fft_x) + fft_x)

        return self.output

    def backward(self, incoming_gradient: NDArray):
        grad = self.norm_b.backward(incoming_gradient)
        # residual connections
        grad_fc_out = grad
        grad_skip_b = grad
        # ---- fully connected ----
        grad = self.fc.backward(grad_fc_out)
        # accumulate skip connection
        grad = grad + grad_skip_b

        grad = self.norm_a.backward(grad)
        # second residual connections
        grad_fft = grad
        grad_skip_a = grad
        # ---- FFT Layer ----
        grad = self.fftlayer.backward(grad_fft)

        # accumulate skip connection
        grad = grad + grad_skip_a

        self.gradient = grad
        return grad

    def __call__(self, x_data: NDArray):
        return self.forward(x_data)

    def purge(self):
        self.norm_a.purge()
        self.fc.purge()
        self.norm_b.purge()

    def get_weights(self) -> tuple[NDArray]:
        return (self.norm_a.get_weights(), self.fc.get_weights(), self.norm_b.weights())

    def get_gradients(self) -> dict[str, NDArray] | None:
        return {
            "norm_a": self.norm_a.get_gradients(),
            "fc": self.fc.get_gradients(),
            "norm_b": self.norm_b.get_gradients(),
        }

    def zero_gradients(self):
        pass

    @property
    def num_parameters(self) -> int:
        return (
            self.norm_a.num_parameters
            + self.fc.num_parameters
            + self.norm_b.num_parameters
        )

    def update_weights(
        self,
        norm_a: dict[str, NDArray],
        fc: dict[str, NDArray],
        norm_b: dict[str, NDArray],
    ) -> None:
        self.norm_a.update_weights(**norm_a)
        self.fc.update_weights(**fc)
        self.norm_b.update_weights(**norm_b)


if __name__ == "__main__":
    from ml_tools.models.optimizers import SGD
    from ml_tools.models.model_loss import MSELoss
    import matplotlib.pyplot as plt

    def make_dataset(n_samples=128, signal_len=128, f1=3, f2=11):
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

    x, y = make_dataset(50, 128, 3, 11)
    loss = MSELoss()
    optimizer = SGD(2e-4)

    attn = FourierAttention(ni=128, no=128, use_2d=True)
    # attn2 = FourierAttention(ni=512, no=512, use_2d=True)
    # attn3 = FourierAttention(ni=512, no=512, use_2d=True)

    all_loss = []
    for _ in range(5000):
        out = attn.forward(x)
        # out = attn3(attn2(attn(x)))
        _l = loss.forward(out, y)
        if _ % 10 == 0:
            print(_l.item())
        all_loss.append(_l.item())
        grad = loss.backward()
        # attn.backward(attn2.backward(attn3.backward(grad)))
        # optimizer.step([attn, attn2, attn3])
        attn.backward(grad)
        optimizer.step([attn])

    plt.plot(all_loss)
    plt.show()
