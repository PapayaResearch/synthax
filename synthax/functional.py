import jax
import jax.numpy as jnp
import chex
from synthax.types import Signal


def midi_to_hz(midi: chex.Array) -> chex.Array:
    """
    Convert from midi (linear pitch) to frequency in Hz.

    Args:
        midi (TODO): Linear pitch on the MIDI scale.

    Return:
        Frequency in Hz.
    """
    return 440.0 * (jnp.exp2((midi - 69.0) / 12.0))


def fix_length(signal: Signal, length: int) -> Signal:
    """
    Pad or truncate (TODO) to specified length.
    """
    num_samples = signal.shape[1]
    if num_samples < length:
        signal = jnp.pad(signal, (0, length - num_samples))
    elif num_samples > length:
        signal = signal[:, :length]
    return signal


def normalize_if_clipping(signal: Signal) -> Signal:
    """
    Only normalize invidiaul signals in batch that have samples
    less than -1.0 or greater than 1.0
    """
    max_sample = jnp.max(jnp.abs(signal), axis=1, keepdims=True)[0]
    return jnp.where(max_sample > 1.0, signal / max_sample, signal)


def normalize(signal: Signal) -> Signal:
    """
    Normalize every individual signal in batch.
    """
    max_sample = jnp.max(jnp.abs(signal), axis=1, keepdims=True)[0]
    return signal / max_sample