from scipy.io import wavfile
import numpy as np
from ml_tools.utilities import rolling_windows_nd, standardize_data
from ml_tools.models.layers.layers import (
    FrequencyFFT,
    FullyConnectedLayer,
    DropoutLayer,
    NormalizeLayer
)
from ml_tools.models.embedding.positional import RopeEmbedding
import matplotlib.pyplot as plt


sample_rate, waveform = wavfile.read("/Users/davidanderson/Dropbox/CODE/tools/ml-tools/test.wav")

waveform_l, waveform_r = standardize_data(waveform[:, 0]), standardize_data(waveform[:, 1])
num_samples = waveform.shape[0]

xs = np.arange(0, num_samples)
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(xs, waveform_l, color='blue')
plt.subplot(2,1,2)
plt.plot(xs, waveform_r, color='red')
plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()

time_per_sample = 1000/ sample_rate
print(time_per_sample, "ms per sample step")

# set window size as ms length ---
window_ms = 20

samples_per_window = int(window_ms / time_per_sample)
print(f"Each {window_ms} ms window will be composed of {samples_per_window} samples")

# now we take windows --
windowed_data = rolling_windows_nd(
    data=waveform_l,
    window_size=samples_per_window,
    num_overlap=samples_per_window // 3,
    axis=0
)

# window count, sample_steps
num_windows, _samp = windowed_data.shape
assert _samp == samples_per_window

#  FFT will output a shape of n//2 + 1.
real_positive_fft_shape = (samples_per_window // 2) + 1

# project dimensions down to half - this should be enough to keep our calculations
# limited without losing too much grainularity in the data.
LATENT_DIMENSION = int(real_positive_fft_shape * 0.75)


# AMPLITUDE ------------------------------------------------------------------
# FUNNEL PROJECTION -->
# dimensional reduction in Feed Forward, built of 1-4 stacked layers
# this gives us a weighted amplitude
# Project down to real_positive_fft_shape? OR LATENT_DIMENSION
# (1920, 1080,   real_positive_fft_shape)
amplitude_layer_a = FullyConnectedLayer(
    ni=samples_per_window,
    no=1200,
    activation_type="swish",
    is_output=False
)
amplitude_layer_b = FullyConnectedLayer(
    ni=1200,
    no=real_positive_fft_shape,
    activation_type="swish",
    is_output=False
)
amplitude_layer_c = FullyConnectedLayer(
    ni=real_positive_fft_shape,
    no=LATENT_DIMENSION,
    activation_type="tanh",
    is_output=True
)

_amp_f = amplitude_layer_c(
    amplitude_layer_b.forward(
        amplitude_layer_a.forward(
            windowed_data
        )
    )
)
print(f"forward pass is shaped: {_amp_f.shape} -> (num_windows, real_positive_fft_shape)")


# FREQUENCY -----------------------------------------------------------------
# real positive only FFT: np rfft process -> final shape of FFT transform is  n//2 + 1.
# transformed_data = np.fft.rfft(windowed_data, axis=-1)
freq_fft_layer = FrequencyFFT(max_sequence_length=500, window_size=samples_per_window)
freq_fc_layer = FullyConnectedLayer(
    ni=real_positive_fft_shape,
    no=LATENT_DIMENSION,
    activation_type="tanh",
    is_output=True
)
_fft = freq_fc_layer(
    freq_fft_layer(windowed_data)
)

# Feed Forward, built of 1-3 stacked layers?
# Project down to LATENT_DIMENSION

stacked = np.hstack((_amp_f, _fft))
print(f"{stacked} shape")

MAX_POSSIBLE_WINDOWS=5000

# Rotary Positional embedding - (relative position and distances -------------
rope_layer = RopeEmbedding(sequence_length=MAX_POSSIBLE_WINDOWS, embedding_dimension=2*LATENT_DIMENSION)
encoded_fin = rope_layer(stacked)

print(encoded_fin.shape)

# Final Aggregation ----------------------------------------------------------
# FINALLY, we will:
#     1) stack FREQUENCY and AMPLITUDE
#     2) ADD ROPE
#     3) end of feature encoder -- then we pass into FEED FORWARD:
#     4) ???? should we do a layer norm?


# feed froward--
FC_IN = 2*LATENT_DIMENSION
dropout_a = DropoutLayer(ni=FC_IN, no=FC_IN)
fc_a = FullyConnectedLayer(
    ni=FC_IN,
    no=1200,
    activation_type="swish",
    is_output=False
)
fc_b = FullyConnectedLayer(
    ni=1200,
    no=900,
    activation_type="swish",
    is_output=False
)
fc_c = FullyConnectedLayer(
    ni=900,
    no=720,
    activation_type="tanh",
    is_output=True
)
norm_a = NormalizeLayer(ni=720, shift_scale=True)
dropout_b = DropoutLayer(ni=720, no=720)


# Pre-dropout ============================
# FC layers   ============================
# layer NORM  ============================
# Dropout     ============================


embedding = dropout_b(norm_a(fc_c(fc_b(fc_a(dropout_a(encoded_fin))))))
print(embedding.shape)
# And Finally, we're at the latent space embedding ------------------------
# NOW WE CAN SEND THE LATENT EMBEDDING INTO our Kernel FourierNetwork.

# the feature encoding and Kernel FourierNetwork layers out into the model's embedding - latent space.
# From here, we can decoder.
