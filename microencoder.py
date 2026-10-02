#!/usr/bin/env python3
"""
DNA sequence convolutional autoencoder — CLI version.

Gaurav Sablok gsablok@proton.me

Reads DNA sequences from a plain-text file (one sequence per line) or a
FASTA file, pads/truncates them to a fixed length, one-hot encodes them,
trains a 1D-convolutional autoencoder, and writes out the learned latent
vectors.

Usage examples
--------------
    python dna_autoencoder_cli.py --input sequences.txt --max-len 64
    python dna_autoencoder_cli.py -i seqs.fasta --fasta --epochs 50 \\
        --batch-size 4 --latent-dim 8 --output latents.csv --save-model model.keras
"""

import sys

import click
import numpy as np
import keras
from keras import layers

# Bases plus "N" for padding / unknown positions. Padding must be part of
# the mapping or one_hot_encode() will KeyError on padded sequences.
BASE_MAPPING = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def read_sequences_plain(path):
    """One sequence per non-empty line."""
    sequences = []
    with open(path) as f:
        for line in f:
            line = line.strip().upper()
            if line:
                sequences.append(line)
    return sequences


def read_sequences_fasta(path):
    """Minimal FASTA parser: lines starting with '>' are headers."""
    sequences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current).upper())
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current).upper())
    return sequences


def pad_sequence(sequence, max_len):
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + "N" * (max_len - len(sequence))


def prepare_sequences(sequences, max_len):
    return [pad_sequence(seq, max_len) for seq in sequences]


def one_hot_encode(sequence):
    try:
        return np.array([BASE_MAPPING[base] for base in sequence], dtype=np.float32)
    except KeyError as e:
        raise ValueError(
            f"Unrecognized base {e} — sequences must contain only A/C/G/T/N."
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_autoencoder(sequence_length, latent_dim):
    autoencoder = keras.Sequential(
        [
            # ================================================
            # Encoder
            # ================================================
            layers.Input(shape=(sequence_length, 4)),
            layers.Conv1D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.Conv1D(filters=16, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dense(latent_dim, name="latent"),

            # ================================================
            # Decoder
            # ================================================
            layers.Dense((sequence_length // 4) * 16, activation="relu"),
            layers.Reshape((sequence_length // 4, 16)),
            layers.UpSampling1D(2),
            layers.Conv1D(filters=16, kernel_size=3, padding="same", activation="relu"),
            layers.UpSampling1D(2),
            layers.Conv1D(filters=4, kernel_size=3, padding="same", activation="softmax"),
        ],
        name="dna_autoencoder",
    )

    autoencoder.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return autoencoder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-i", "--input", "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a file of DNA sequences (plain text, one per line, unless --fasta is set).",
)
@click.option(
    "--fasta", is_flag=True, default=False,
    help="Parse --input as FASTA instead of plain text.",
)
@click.option(
    "--max-len", type=int, default=None,
    help="Pad/truncate all sequences to this length. Defaults to the longest sequence found.",
)
@click.option(
    "--latent-dim", type=int, default=4, show_default=True,
    help="Size of the bottleneck / latent representation.",
)
@click.option(
    "--epochs", type=int, default=100, show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size", type=int, default=2, show_default=True,
    help="Training batch size.",
)
@click.option(
    "--output", "output_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Where to write the latent vectors as CSV. If omitted, prints to stdout.",
)
@click.option(
    "--save-model", "model_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional path to save the trained autoencoder (e.g. model.keras).",
)
@click.option(
    "--quiet", is_flag=True, default=False,
    help="Suppress Keras training/progress output.",
)
def main(input_path, fasta, max_len, latent_dim, epochs, batch_size, output_path, model_path, quiet):
    """Train a convolutional autoencoder on DNA sequences and export latent vectors."""

    reader = read_sequences_fasta if fasta else read_sequences_plain
    sequences = reader(input_path)

    if not sequences:
        click.echo(f"No sequences found in {input_path!r}.", err=True)
        sys.exit(1)

    if max_len is None:
        max_len = max(len(s) for s in sequences)
        # Round up to a multiple of 4 so the encoder's two stride-2 Conv1D
        # layers and the decoder's two UpSampling1D layers line up exactly.
        remainder = max_len % 4
        if remainder != 0:
            max_len += 4 - remainder
        click.echo(f"--max-len not given; using {max_len} (padded to a multiple of 4).")
    elif max_len % 4 != 0:
        click.echo("Error: --max-len must be a multiple of 4 for this architecture.", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(sequences)} sequences.")

    padded = prepare_sequences(sequences, max_len)
    try:
        X = np.array([one_hot_encode(seq) for seq in padded], dtype=np.float32)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    sequence_length = X.shape[1]
    click.echo(f"Encoded array shape: {X.shape}")

    autoencoder = build_autoencoder(sequence_length, latent_dim)
    if not quiet:
        autoencoder.summary()

    autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0 if quiet else 1,
    )

    encoder = keras.Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer("latent").output,
    )
    latent_vectors = encoder.predict(X, verbose=0)

    click.echo(f"Latent vectors shape: {latent_vectors.shape}")

    if output_path:
        np.savetxt(output_path, latent_vectors, delimiter=",")
        click.echo(f"Latent vectors written to {output_path}")
    else:
        for row in latent_vectors:
            click.echo(",".join(f"{v:.6f}" for v in row))

    if model_path:
        autoencoder.save(model_path)
        click.echo(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()