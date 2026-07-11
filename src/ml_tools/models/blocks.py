from numpy.typing import NDArray
import numpy as np
from ml_tools.models.activations import mod_relu, mod_relu_derivative
from ml_tools.models.layers.layers import (
    Layer,
    FullyConnectedLayer,
    FourierLayer,
    NormalizeLayer,
)

EPSILON = 1e-15


class FourierAttention(Layer):
    def __init__(self, ni: int, no: int, use_2d: bool = False):
        super().__init__()
        self.fftlayer = FourierLayer(use_2d, fft_axis=1)
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
        return (
            self.norm_a.get_weights(),
            self.fc.get_weights(),
            self.norm_b.weights()
        )

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


class SpectreAttention(Layer):
    """
    Spectre network FFT attention
    https://arxiv.org/pdf/2502.18394

    """
    def __init__(self, sequence_length: int, hidden_dim: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        self.num_frequencies = sequence_length // 2 + 1
        self.freq_idx = np.arange(self.num_frequencies)

        # TODO: adapt the activation bias to a learnable parameter
        self.activation_bias = np.zeros(self.num_frequencies) - 0.1

        self.fc_query = FullyConnectedLayer(ni=self.hidden_dim, no=self.hidden_dim, activation_type="linear")
        self.fc_values = FullyConnectedLayer(ni=self.hidden_dim, no=self.hidden_dim, activation_type="linear")

        self.fc_1 = FullyConnectedLayer(ni=self.hidden_dim, no=self.hidden_dim, activation_type="relu")
        self.fc_2 = FullyConnectedLayer(ni=self.hidden_dim, no=2*self.num_frequencies, activation_type="linear")

        # LayerNorm on global descriptor (paper Section 3.2, Step 3a: q_bar = LN(1/n * sum_i q_i))
        self.layer_norm_q = NormalizeLayer(ni=self.hidden_dim, shift_scale=True)

        self.activation: callable = mod_relu
        self.activation_derivative: callable =  mod_relu_derivative

    def forward(self, input_data: NDArray):
        """
        input_data: (batch, sequence, hidden)
        """
        self.input = input_data
        num_samples, self.num_windows, num_steps = input_data.shape

        # projections
        query_forward = self.fc_query(input_data)
        value_forward = self.fc_values(input_data)

        # FFT along SEQUENCE axis
        value_transform = np.fft.rfft(value_forward, n=self.sequence_length, axis=1, norm="ortho")

        # global descriptor: mean over sequence, then LayerNorm with learnable gamma/beta
        # paper Section 3.2, Step 3a: q_bar = LN(1/n * sum_i q_i)
        seq_mu = np.mean(query_forward, axis=1)
        global_descriptor = self.layer_norm_q(seq_mu)

        # forward pass through weights to create the "complex gate"
        gate_raw = self.fc_2(self.fc_1(global_descriptor))

        g_real, g_imag = np.split(gate_raw, 2, axis=-1)
        self.gate_raw = g_real + 1j * g_imag

        # mod-ReLU for complex numbers
        self.gate = self.activation(self.gate_raw, self.activation_bias)

        # diagonal spectral gating
        # NOTE: positional phase e^{j2πkt/N} is decode-only (Algorithm 1 DECODESTEP); omitted for batch training
        self.values_gated = value_transform * self.gate[..., None]

        # invert the fourier transform
        self.output = np.fft.irfft(self.values_gated, n=self.sequence_length, axis=1, norm="ortho")[:, :self.num_windows, :]
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        # invert the output transform to put us back in frequency domain
        batch_size, sequence_length, hidden_dim = incoming_gradient.shape

        # ---- inverse FFT backward ----
        dvalues_gated = np.fft.rfft(incoming_gradient, axis=1, n=self.sequence_length, norm="ortho")

        # spectral gating backward
        v_hat = np.fft.rfft(self.fc_values.output.reshape(self.fc_values.in_shape), axis=1, n=self.sequence_length, norm="ortho")

        dv_hat = dvalues_gated * np.conj(self.gate)[..., None]
        dgate = np.sum(dvalues_gated * np.conj(v_hat), axis=2) / hidden_dim

        # mod_relu backward (no positional phase in batch training)
        self.grad_bias, dg_complex = self.activation_derivative(
            self.gate_raw, self.activation_bias, dgate
        )

        # ---- complex and real split ----
        dg_real = dg_complex.real
        dg_imag = dg_complex.imag
        dg_raw = np.concatenate([dg_real, dg_imag], axis=-1)

        # Fully connected
        dh = self.fc_2.backward(dg_raw)
        dglobal_descriptor = self.fc_1.backward(dh)

        # LayerNorm backward (learnable gamma/beta handled inside NormalizeLayer)
        dseq_mu = self.layer_norm_q.backward(dglobal_descriptor)

        # distribute mean-gradient across sequence
        dq = np.repeat(dseq_mu[:, None, :], sequence_length, axis=1) / sequence_length

        # ---- propagate into fc layers ----
        dinput_from_q = self.fc_query.backward(dq)

        dv_time = np.fft.irfft(dv_hat, n=self.sequence_length, axis=1, norm="ortho")[:, :self.num_windows, :]
        dinput_from_v = self.fc_values.backward(dv_time)

        # ---- combine input gradients ----
        return dinput_from_q + dinput_from_v

    def get_gradients(self) -> dict[str, NDArray|dict] | None:
        return {
            "grad_bias": self.grad_bias,
            "fc_query": self.fc_query.get_gradients(),
            "fc_values": self.fc_values.get_gradients(),
            "fc_1": self.fc_1.get_gradients(),
            "fc_2": self.fc_2.get_gradients(),
            "layer_norm_q": self.layer_norm_q.get_gradients(),
        }

    def purge(self) -> None:
        self.layer_norm_q.purge()

    def zero_gradients(self):
        pass

    @property
    def num_parameters(self) -> int:
        return (
            self.activation_bias.size
            + self.fc_query.num_parameters
            + self.fc_values.num_parameters
            + self.fc_1.num_parameters
            + self.fc_2.num_parameters
            + self.layer_norm_q.scale_gamma.size
            + self.layer_norm_q.shift_beta.size
        )

    def update_weights(
            self,
            grad_bias: NDArray,
            fc_query: dict[str, NDArray],
            fc_values: dict[str, NDArray],
            fc_1: dict[str, NDArray],
            fc_2: dict[str, NDArray],
            layer_norm_q: dict[str, NDArray],
    ) -> None:
        self.activation_bias -= grad_bias
        self.fc_query.update_weights(**fc_query)
        self.fc_values.update_weights(**fc_values)
        self.fc_1.update_weights(**fc_1)
        self.fc_2.update_weights(**fc_2)
        self.layer_norm_q.update_weights(**layer_norm_q)



class LatentSpectralAttention(SpectreAttention):
    def __init__(self,
                 d_model=128 * 128,
                 sequence_length=512,
                 num_heads=128,
                 q_latent_dim=12,
                 kv_latent_dim=4):
        """
        LoRA Latent attention and Spectral mixing along sequences
        Dimension reprojection in Sequence dimension using Spectral mixing
        Dimension reprojection in latent / hidden dimension with LoRA

        Parameters
        ----------
        d_model :
        num_heads :
        q_latent_dim :
        kv_latent_dim :
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim
        self.head_dim = d_model // num_heads
        self.sequence_length = sequence_length


        self.frequency_layer = SpectreLayer()

        self.Wq_d = FullyConnectedLayer(d_model, q_latent_dim, "linear")
        self.Wkv_d = FullyConnectedLayer(d_model, kv_latent_dim, "linear")

        # TODO: just started this - Spectre has Query and Value weights
        # Query
        self.Wq_d = nn.Linear(d_model, q_latent_dim)

        # Precomputed matrix multiplications of W_q^U and W_k^U, for multiple heads
        self.W_qk = nn.Linear(q_latent_dim, num_heads * kv_latent_dim)
        # Key/Value latent projections
        self.Wkv_d = nn.Linear(d_model, kv_latent_dim)
        self.Wv_u = nn.Linear(kv_latent_dim, num_heads * head_dim)
        # Output projection
        self.Wo = nn.Linear(num_heads * head_dim, d_model)


    def forward(self, input_data: NDArray) -> NDArray:
        batch_size, seq_len, d_model = x.shape

        query_forward = self.fc_query(input_data)
        value_forward = self.fc_values(input_data)

        # FFT along SEQUENCE axis
        value_transform = np.fft.rfft(value_forward, axis=1)

        # norm over the sequence axis
        seq_mu = np.mean(query_forward, axis=1)
        global_descriptor = (seq_mu - np.mean(seq_mu)) / (np.std(seq_mu) + EPSILON)

        # forward pass through weights to create the "complex gate"
        gate_raw = self.fc_2(self.fc_1(global_descriptor))

        g_real, g_imag = np.split(gate_raw, 2, axis=-1)
        self.gate_raw = g_real + 1j * g_imag

        # mod-Relu for complex numbers
        self.gate = self.activation(self.gate_raw, self.activation_bias)

        # positional phase in the frequency domains
        phase = np.exp(1j * 2 * np.pi * self.freq_idx / self.sequence_length)
        self.gate *= phase[None, :]

        # diagonal spectral gating
        # self.values_gated = value_transform * self.gate[:, :]
        self.values_gated = value_transform * self.gate[..., None]

        # invert the fourier transform
        self.output = (
            np.fft.irfft(self.values_gated, n=self.sequence_length, axis=1)
            / self.sequence_length
        )



        # --------------------------------------------------------------------
        # Attention score, shape: (batch_size, num_heads, seq_len, seq_len)
        C_qW_qk = self.W_qk(C_q).view(
            batch_size, seq_len, self.num_heads, self.kv_latent_dim
        )
        scores = torch.matmul(
            C_qW_qk.transpose(1, 2), C_kv.transpose(-2, -1)[:, None, ...]
        ) / math.sqrt(self.kv_latent_dim)

        # Attention computation
        attn_weight = torch.softmax(scores, dim=-1)
        # Restore V from latent space
        V = self.Wv_u(C_kv).view(batch_size, seq_len, self.num_heads, -1)
        # Compute attention output, shape: (batch_size, seq_len, num_heads, head_dim)
        output = (
            torch.matmul(attn_weight, V.transpose(1, 2)).transpose(1, 2).contiguous()
        )
        # Concatentate the heads, then apply output projection
        output = self.Wo(output.view(batch_size, seq_len, -1))
        return output


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Diagnostic training loop for SpectreAttention.
    #
    # Task  : identity mapping  y = x  on multi-frequency sine waves.
    # Dims  : kept small (batch=4, seq=16, hidden=16) so one pass is fast and
    #         gradient overflows are immediately visible.
    # Checks: gradient NaN / Inf detection, weight-update magnitudes, and
    #         whether the loss strictly decreases over 1 000 steps.
    # -----------------------------------------------------------------------
    from ml_tools.models.optimizers import SGD
    from ml_tools.models.model_loss import MSELoss

    np.random.seed(42)

    BATCH  = 4
    SEQ    = 16
    HIDDEN = 16
    # MSELoss now divides by all elements (BATCH*SEQ*HIDDEN = 1024), making
    # per-element gradients ~256x smaller than before.  Compensate here.
    LR     = 0.5
    STEPS  = 1000

    # Build a fixed dataset: x[b, t, d] = sin(freq[d]*t + phase[b,d])
    rng    = np.random.RandomState(0)
    t_axis = np.linspace(0, 2 * np.pi, SEQ, endpoint=False)
    freqs  = np.arange(1, HIDDEN + 1, dtype=float)
    phases = rng.uniform(0, 2 * np.pi, (BATCH, HIDDEN))
    x = np.sin(
        freqs[None, None, :] * t_axis[None, :, None] + phases[:, None, :]
    ).astype(np.float32)
    y = x.copy()   # identity target

    model   = SpectreAttention(sequence_length=SEQ, hidden_dim=HIDDEN)
    loss_fn = MSELoss()
    opt     = SGD(LR)

    # ---- helper: largest absolute gradient across weight tensors -----------
    def _max_grad(grads_dict, key):
        v = grads_dict.get(key)
        if isinstance(v, dict):
            vals = [arr for arr in v.values() if arr is not None]
            return float(np.max([np.max(np.abs(a)) for a in vals])) if vals else float("nan")
        return float(np.max(np.abs(v))) if v is not None else float("nan")

    def _has_nan_or_inf(grads_dict):
        for v in grads_dict.values():
            if isinstance(v, dict):
                for arr in v.values():
                    if arr is not None and (np.any(np.isnan(arr)) or np.any(np.isinf(arr))):
                        return True
            elif v is not None:
                if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                    return True
        return False

    # ---- training loop -----------------------------------------------------
    print(f"{'step':>6}  {'loss':>10}  {'|gW_q|':>8}  {'|gW_v|':>8}  {'|g_bias|':>9}  status")
    print("-" * 62)

    losses = []
    stop   = False
    for step in range(STEPS + 1):
        out  = model.forward(x)
        loss = loss_fn.forward(out, y)
        losses.append(float(loss))

        grad_out = loss_fn.backward()
        model.backward(grad_out)

        if step % 100 == 0 or stop:
            g    = model.get_gradients()
            gq   = _max_grad(g, "fc_query")
            gv   = _max_grad(g, "fc_values")
            gb   = _max_grad(g, "grad_bias")
            bad  = _has_nan_or_inf(g)
            div  = len(losses) > 1 and losses[-1] > losses[0] * 100
            status = "NaN/Inf" if bad else ("diverging" if div else "ok")
            print(f"{step:>6}  {loss:>10.6f}  {gq:>8.5f}  {gv:>8.5f}  {gb:>9.6f}  {status}")
            if bad:
                print("  → stopping: numerical error detected.")
                stop = True

        if stop or step == STEPS:
            break

        opt.step([model])

    # ---- summary -----------------------------------------------------------
    ratio = losses[-1] / (losses[0] + 1e-30)
    print()
    print(f"Initial loss : {losses[0]:.6f}")
    print(f"Final loss   : {losses[-1]:.6f}")
    print(f"Ratio        : {ratio:.4f}  (target < 0.50)")
    print(f"Converged    : {ratio < 0.50}")
