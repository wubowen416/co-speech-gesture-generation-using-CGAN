import torch
from models import Generator


def load_model(param_path):

    input_size = 6
    noise_size = 10
    hidden_size = 128
    out_size = 33

    # Load parameters
    G = Generator(i_size=input_size, n_size=noise_size,
                  o_size=out_size, h_size=hidden_size)
    G.load_state_dict(torch.load(param_path))
    return G
