import math
from typing import Tuple, Sequence

from flax import linen as nn
from flax.linen.initializers import lecun_normal
from flax.linen.initializers import constant, orthogonal

from conf.config import EncoderConfig, BertTrainConfig
import numpy as np
import jax
import jax.numpy as jnp

from models import crop_arf_vrf


class SelfAttention(nn.Module):
    hidden_size: int
    num_heads: int
    output_size: int
    dropout_rate: float = 0.1
    use_bias: bool = True
    broadcast_dropout: bool = True

    def setup(self):
        self.attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.output_size,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_init=lecun_normal(),
            broadcast_dropout=self.broadcast_dropout
        )

    @nn.compact
    def __call__(self, x, deterministic: bool = True, train: bool = True):
        # SelfAttention layer def
        return self.attn(x, deterministic=deterministic)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) implemented using JAX and Flax.
    """

    num_layers: int
    hidden_size: int
    output_size: int
    activation: str = "relu"
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.array, train: bool = True):
        """
        Forward pass of the MLP.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, input_dim).

        Returns:
            jnp.ndarray: Output of the MLP.
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = activation(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Output Layer
        x = nn.Dense(
            self.output_size, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(x)

        return x


class MLP_VAE(nn.Module):
    """
    Multi-Layer Perceptron (MLP) base VAE encoder implemented using JAX and Flax.
    """

    num_layers: int
    hidden_size: int
    output_size: int
    activation: str = "relu"
    dropout_rate: float = 0.1

    def reparameterize(self, rng:jax.random.PRNGKey, mu: jnp.ndarray, logvar: jnp.ndarray, deterministic: bool = False):
        """
        Reparameterization trick: Sample z ~ N(mu, sigma^2)
        """

        def stochastic_fn(_):
            std = jnp.exp(0.5 * jnp.clip(logvar, -10, 10))  # Prevent overflow
            eps = jax.random.normal(rng, mu.shape)
            return mu + eps * std

        return jax.lax.cond(deterministic, lambda _: mu, stochastic_fn, operand=None)

    @nn.compact
    def __call__(self, x: jnp.array, rng:jax.random.PRNGKey, train: bool = True, deterministic: bool = False):

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        """
        Forward pass of the MLP.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, input_dim).
            key(jax.random.PRNGKey): random key for reparameterization
            deterministic(bool): whether to use deterministic mode

        Returns:
            jnp.ndarray: Output of the MLP.
        """
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = activation(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        mu = nn.Dense(self.output_size)(x)
        log_var = nn.Dense(self.output_size)(x)

        z = self.reparameterize(rng, mu, log_var, deterministic)

        return z, mu, log_var


class MeanPool(nn.Module):
    """
    FlaxBERT model with mean pooling and a single-layer MLP.
    """
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, text_embeddings, train: bool = True, deterministic: bool = False):
        """
        Forward pass of the FlaxBERT model with mean pooling.

        Args:
            text_embeddings (jnp.ndarray): BERT output of shape (batch_size, seq_len, hidden_size).

        Returns:
            jnp.ndarray: Final representation of shape (batch_size, mlp_output_dim).
        """
        # Apply mean pooling on BERT output
        text_representation = jnp.mean(text_embeddings, axis=1, keepdims=True)  # (batch_size, 1, hidden_size)

        mlp_output = nn.Dense(self.output_size)(text_representation)

        # Apply tanh activation separately
        mlp_output = jnp.tanh(mlp_output)

        return mlp_output


class ConvForward(nn.Module):
    output_dim: Sequence[int]
    act_shape: Tuple[int, int]
    arf_size: int
    vrf_size: int
    hidden_dims: Tuple[int]

    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        h1, h2 = self.hidden_dims


        act, critic = crop_arf_vrf(map_x, self.arf_size, self.vrf_size)

        act = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(act)
        act = activation(act)
        act = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(act)
        act = activation(act)

        act = act.reshape((act.shape[0], -1))
        act = jnp.concatenate((act), axis=-1)

        act = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(act)
        act = activation(act)


        act = nn.Dense(
            self.output_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        act = act.reshape((1, -1))

        return act



def apply_encoder_model(config: EncoderConfig, dropout_rate: float = 1.0, broadcast_dropout: bool = True):
    if config.model == 'mlp':
        encoding_model = MLP(num_layers=config.num_layers,
                             hidden_size=config.hidden_dim,
                             output_size=config.output_dim,
                             dropout_rate=dropout_rate)
    elif config.model == 'sa':
        encoding_model = SelfAttention(hidden_size=config.hidden_dim,
                                       num_heads=config.num_heads,
                                       output_size=config.output_dim,
                                       dropout_rate=dropout_rate,
                                       broadcast_dropout=broadcast_dropout)
    elif config.model == 'mp':
        encoding_model = MeanPool(hidden_size=config.hidden_dim,
                                  output_size=config.output_dim)
    elif config.model == 'mlp_vae':
        encoding_model = MLP_VAE(num_layers=config.num_layers,
                             hidden_size=config.hidden_dim,
                             output_size=config.output_dim,
                                 dropout_rate=dropout_rate
                                 )
    else:
        raise ValueError(f"Model {config.model} not supported")

    return encoding_model


def apply_decoder_model(config, dropout_rate: float = 1.0):
    model = MLP(num_layers=config.num_layers,
                hidden_size=config.hidden_dim,
                output_size=config.output_dim,
                dropout_rate=dropout_rate)
    return model


class apply_only_decoder_model(nn.Module):
    config: BertTrainConfig
    def setup(self) -> None:
        self.decoder = apply_decoder_model(self.config)

        self.conv = ConvForward(output_dim=512, act_shape=(1, 1), arf_size=1, vrf_size=1, hidden_dims=(1, 1))

    @nn.compact
    def __call__(self, x):
        x = self.conv(x)

        x = self.decoder(x)
        return x



class apply_model(nn.Module):

    config: BertTrainConfig
    activation: str = "relu"

    def setup(self) -> None:
        self.model = self.config.encoder.model

        self.encoder = apply_encoder_model(self.config.encoder, self.config.dropout_rate, self.config.broadcast_dropout)
        self.decoder = apply_decoder_model(self.config.decoder, self.config.dropout_rate)

        self.deterministic = self.config.deterministic
        self.rng = jax.random.PRNGKey(self.config.seed)

    @nn.compact
    def __call__(self, x, rng, sampled_buffer=None, is_train=True):
        outputs = {
            'z': None,
            'logits': None,
            'mu': jnp.zeros((x.shape[0], self.encoder.output_size)),
            'log_var': jnp.zeros((x.shape[0], self.encoder.output_size))
        }

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh


        if self.model == 'sa':
            z = self.encoder(x, deterministic=not is_train)

        elif self.model == 'mlp_vae':
            deterministic = jax.lax.cond(
                is_train,
                lambda _: False,  # If is_train=True, deterministic=False
                lambda _: True,   # If is_train=False, deterministic=True
                operand=None
            )
            z, mu, log_var = self.encoder(x, rng, deterministic=deterministic, train=is_train)
            outputs['mu'] = mu
            outputs['log_var'] = log_var
        else:
            z = self.encoder(x, train=is_train)

        outputs['z'] = z

        # Concatenate
        if sampled_buffer is not None:

            sampled_buffer, _ = crop_arf_vrf(sampled_buffer, self.config.arf_size, self.config.vrf_size)


            for _ in range(self.config.decoder.num_layers):

                state_x = nn.Conv(
                    features=self.decoder.hidden_size // 2, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0)
                )(sampled_buffer)
                state_x = activation(state_x)
                state_x = nn.Dropout(rate=self.config.dropout_rate)(state_x, deterministic=not is_train)

            state_x = state_x.reshape((state_x.shape[0], -1))

            state_x = nn.Dense(self.decoder.hidden_size,
                           kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(state_x)
            state_x = activation(state_x)
            state_x = nn.Dropout(rate=self.config.dropout_rate)(state_x, deterministic=not is_train)

            z = jnp.concatenate([z, state_x], axis=1)

        logits = self.decoder(x=z, train=is_train)

        outputs['logits'] = logits
        return outputs
