#!/usr/bin/env python3
"""Convert NVIDIA neuralappearance checkpoints to a Three.js runtime asset.

The official training code writes Slang/Falcor-oriented checkpoints containing
`model.json` and multi-channel latent EXR files. This adapter extracts the
subset supported by `NeuralAppearanceNodeMaterial`: an 8-channel latent texture,
two learned shading frames, and a direct-light BRDF decoder with ReLU/linear
hidden activations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "three-neural-appearance"
VERSION = 3
LATENT_CHANNELS = 8
SUPPORTED_INPUT_SIZE = 20
IBL_INPUT_SIZE = 14
IBL_OUTPUT_SIZE = 13


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Checkpoint folder containing model.json")
    parser.add_argument("output", type=Path, help="Output JSON file for Three.js")
    parser.add_argument("--name", default="Converted Neural Appearance Material")
    parser.add_argument("--wrap", choices=("repeat", "clamp"), default="repeat")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    model_path = checkpoint / "model.json"

    with model_path.open("r", encoding="utf8") as f:
        model = json.load(f)

    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "name": args.name,
        "source": "NVlabs/neuralappearance checkpoint",
        "latents": convert_latents(checkpoint, model, args.wrap),
        "outputs": {
            "brdf": convert_decoder(model),
            "ibl": create_default_ibl_head(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def convert_latents(checkpoint: Path, model: dict[str, Any], wrap: str) -> dict[str, Any]:
    latents = model["latents"]
    if latents["num_channels"] != LATENT_CHANNELS:
        raise ValueError(f"Only {LATENT_CHANNELS}-channel latent textures are supported.")

    mip_levels = latents["num_mip_levels"]
    texture_sets = [[], []]

    for mip_level in range(mip_levels):
        image_path = checkpoint / f"latents.material0.mip{mip_level}.exr"
        image = read_exr(image_path)
        if image.shape[-1] != LATENT_CHANNELS:
            raise ValueError(f"{image_path} must contain {LATENT_CHANNELS} channels.")

        for texture_index in range(2):
            first = texture_index * 4
            rgba = image[:, :, first:first + 4].astype(np.float32)
            texture_sets[texture_index].append(
                {
                    "width": int(rgba.shape[1]),
                    "height": int(rgba.shape[0]),
                    "data": flatten(rgba),
                }
            )

    return {
        "channels": LATENT_CHANNELS,
        "wrap": wrap,
        "textures": [
            {"wrap": wrap, "mipmaps": texture_sets[0]},
            {"wrap": wrap, "mipmaps": texture_sets[1]},
        ],
    }


def convert_decoder(model: dict[str, Any]) -> dict[str, Any]:
    decoder = model["decoder"]
    rotation = decoder.get("rotation", {}).get("rotation_decoder", {}).get("network", {})
    rotation_layers = rotation.get("mlp_layers")
    if not rotation_layers or len(rotation_layers) != 1:
        raise ValueError("Expected a single linear rotation decoder layer.")

    rotation_layer = rotation_layers[0]
    if rotation_layer["num_inputs"] != LATENT_CHANNELS or rotation_layer["num_outputs"] != 12:
        raise ValueError("Expected an 8-to-12 rotation decoder for two shading frames.")

    network = decoder["network"]
    layers = []
    for index, layer in enumerate(network["mlp_layers"]):
        activation = "linear" if index == len(network["mlp_layers"]) - 1 else normalize_activation(decoder, index)
        layers.append(
            {
                "inputSize": layer["num_inputs"],
                "outputSize": layer["num_outputs"],
                "activation": activation,
                "weights": float_list(layer["weights"]),
                "biases": float_list(layer["biases"]),
            }
        )

    if layers[0]["inputSize"] != SUPPORTED_INPUT_SIZE:
        raise ValueError(f"Only {SUPPORTED_INPUT_SIZE}-input decoders are supported.")
    if layers[-1]["outputSize"] != 3:
        raise ValueError("Decoder output must be RGB.")

    return {
        "inputSize": SUPPORTED_INPUT_SIZE,
        "rotation": {
            "inputSize": LATENT_CHANNELS,
            "outputSize": 12,
            "weights": float_list(rotation_layer["weights"]),
        },
        "layers": layers,
        "outputActivation": normalize_output_activation(decoder),
    }


def create_default_ibl_head() -> dict[str, Any]:
    return {
        "inputSize": IBL_INPUT_SIZE,
        "layers": [
            {
                "inputSize": IBL_INPUT_SIZE,
                "outputSize": IBL_OUTPUT_SIZE,
                "activation": "linear",
                "weights": [0.0] * (IBL_INPUT_SIZE * IBL_OUTPUT_SIZE),
                "biases": [
                    0.0, 0.0, 1.0,
                    0.5, 0.5, 0.5,
                    0.0, 0.0, 1.0,
                    0.0,
                    0.04, 0.04, 0.04,
                ],
            }
        ],
        "outputActivation": {"type": "linear"},
    }


def normalize_activation(decoder: dict[str, Any], index: int) -> str:
    activations = decoder.get("hidden_activations", "relu")
    if isinstance(activations, str):
        name = activations
    else:
        name = activations[index]

    name = name.lower()
    if name in ("relu", "leakyrelu", "smelu"):
        # The runtime currently supports ReLU-family monotonic hidden layers.
        # Unsupported exact nonlinearities should be added explicitly before
        # using the converter for quality comparisons.
        return "relu"
    if name == "linear":
        return "linear"

    raise ValueError(f"Unsupported hidden activation: {name}")


def normalize_output_activation(decoder: dict[str, Any]) -> dict[str, Any]:
    config = decoder.get("output_activation", {"type": "linear"})
    name = config["type"].lower()

    if name == "exp":
        return {"type": "exp", "offset": float(config.get("offset", 0.0))}
    if name == "scaledsigmoid":
        return {"type": "scaledSigmoid", "scale": float(config.get("scale", 1.0))}
    if name == "linear":
        return {"type": "linear"}

    raise ValueError(f"Unsupported output activation: {config['type']}")


def read_exr(path: Path) -> np.ndarray:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("Install imageio with OpenEXR support to convert latent EXR files.") from exc

    if not path.exists():
        raise FileNotFoundError(path)

    image = iio.imread(path)
    if image.ndim == 2:
        image = image[:, :, None]
    return np.asarray(image)


def flatten(array: np.ndarray) -> list[float]:
    return [float(value) for value in array.reshape(-1)]


def float_list(values: list[Any]) -> list[float]:
    return [float(value) for value in values]


if __name__ == "__main__":
    main()
