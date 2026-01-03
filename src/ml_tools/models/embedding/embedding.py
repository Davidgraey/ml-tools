import numpy as np
from ml_tools.models.layers import Layer
# make embedding layer
# take in variables, catgorical, etc
# Autoencoder -> layers to project to latent space
# reconstruction layer to rebuild the input
# train on reconstruct loss


def fourier_embedding(Layer):
    #  https://ieeexplore.ieee.org/document/9616294
    #
    def forward(x):
        fft_output = np.fft.fft2(x, axes=(-2, -1)).real
        # scale, constrain or norm?

    def backward(incoming_grad):
        df_fft = np.fft.ifft(incoming_grad).real

        # Gemma's sumary - suspect!! --
        # fft_X = np.fft.fft2(X, s=X.shape)
        # fft_dL_dY = np.fft.fft2(dL_dY, s=dL_dY.shape)
        #
        # # 2. Element-wise multiplication in the frequency domain
        # # The complex conjugate of the input FFT is typically used in the derivation for the weight gradient
        # fft_dL_dW = np.conjugate(fft_X) * fft_dL_dY
        # inverse
        # dL_dW = np.fft.ifft2(fft_dL_dW)
        # dL_dW = np.real(dL_dW)[: W.shape[0], : W.shape[1]]

        # 1. Flip the kernel (weights) by 180 degrees
        # W_flipped = np.rot180(W, 2)
        # # 2. Forward FFT of the flipped kernel and the backpropagated error dL_dY
        # fft_W_flipped = np.fft.fft2(
        #     W_flipped, s=dL_dY.shape
        # )  # Pad kernel to error shape
        # fft_dL_dY = np.fft.fft2(dL_dY, s=dL_dY.shape)
        #
        # # 3. Element-wise multiplication in the frequency domain
        # fft_dL_dX = fft_dL_dY * fft_W_flipped
        #
        # # 4. Inverse FFT to get the gradient of the input
        # dL_dX = np.fft.ifft2(fft_dL_dX)
        #
        # # Take the real part and crop any potential extra padding
        # dL_dX = np.real(dL_dX)[: X.shape[0], : X.shape[1]]


if __name__ == "__main__":
    # (num_samples, x_axis (words in sentence, etc), hidden dimension)
    x = np.random.uniform(size=(12, 28, 512))

    x_hat = np.fft.fft2(x, axes=(-2, -1)).real.astype(np.float32)

    assert x.shape == x_hat.shape

    inverted = np.fft.ifft2(x_hat, axes=(-1, -2)).real.astype(np.float32)

    np.isclose(x, inverted)


