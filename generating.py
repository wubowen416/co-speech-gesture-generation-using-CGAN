import argparse
import os
import torch
import numpy as np

from data_processing.prosodic_feature import extract_prosodic_features
from models import Generator
from noise_generator import NoiseGenerator
from torch_dataset import TrainSet
from utils import load_model


def generate_motion(model, inputs, noise=None):

    dset = TrainSet()

    # Config input
    inputs = dset.scale_x(inputs)
    inputs = torch.FloatTensor(inputs).unsqueeze(1)

    if noise is not None:
        assert len(noise) == 3, "shape of noise must be (T, N, noise_dim)"
        assert noise.shape[0] == inputs.shape[0], "time step of noise is not compatible with inputs"
        assert noise.shape[1] == inputs.shape[1], "batch size of noise is not compatible with inputs"
        assert noise.shape[2] == model.n_size, "Noise dim is not compatible with model"
    else:
        noise_g = NoiseGenerator()
        noise = noise_g.gaussian_variating(
            T=inputs.shape[0], F=40, size=model.n_size, allow_indentical=True)

    noise = torch.FloatTensor(noise)
    with torch.no_grad():
        outs = model.forward_given_noise_seq(inputs, noise).squeeze(1).numpy()
    outs = dset.rescale_y(outs)

    return outs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('wav_path', help='Path of input .wav file')
    parser.add_argument(
        'output_path', help='Path for saving .npy file')
    parser.add_argument('-mp', help='Which parameters to import',
                        default='pre_trained_model/unroll_1000.pt')

    args = parser.parse_args()

    G = load_model(args.mp)

    # Configure input
    inputs = extract_prosodic_features(args.wav_path)

    # Generate motions
    outs = generate_motion(G, inputs)

    np.save(args.output_path, outs)
