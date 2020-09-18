from models import Generator, Discriminator
from torch_dataset import TrainSet

import torch
import copy
import numpy as np
import tqdm
import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
sns.set_style('whitegrid')

device = 'cuda'


# * Create dir for store models
try:
    shutil.rmtree('saved_models')
except:
    print('Folder does not exist')
os.makedirs('saved_models', exist_ok=True)


# --- Data params
noise_size = 10
prosody_size = 6
hidden_size = 128
pose_size = 33
d_out_size = 1

# --- Training params
n_epochs = 5
sample_size = 32
d_lr = 1e-4
g_lr = 1e-4
unroll_steps = 10
log_interval = 100


# --- Init dataset
dataset = TrainSet()
dataset.scaling(True)

# --- Init net i_size, n_size, o_size
G = Generator(i_size=prosody_size, n_size=noise_size,
              o_size=pose_size, h_size=hidden_size).to(device)
D = Discriminator(i_size=pose_size, c_size=prosody_size,
                  h_size=hidden_size).to(device)

g_opt = torch.optim.Adam(G.parameters(), lr=g_lr)
d_opt = torch.optim.Adam(D.parameters(), lr=d_lr)

bce_loss = torch.nn.BCELoss()


d_real_loss = []
d_fake_loss = []
g_loss = []


def sample_noise(batch_size, dim):
    return np.random.normal(0, 1, (batch_size, dim))


for epoch in tqdm.tqdm(range(n_epochs)):
    # for i in range(n_epochs):

    torch.cuda.empty_cache()

    # * Random sample
    idxs = np.random.randint(low=0, high=len(dataset), size=sample_size)

    # * Configure real data
    xs = [dataset[i][0] for i in idxs]
    real_ys = [dataset[i][1] for i in idxs]

    # * Generate fake data
    fake_ys = []
    for x in xs:
        x = torch.Tensor(x).unsqueeze(1).to(device)
        noise = torch.Tensor(sample_noise(1, noise_size)
                             ).unsqueeze(1).to(device)
        with torch.no_grad():
            fake_y = G(x, noise)
        fake_ys.append(fake_y)

    # * Train D
    d_opt.zero_grad()
    d_real_error = 0
    d_fake_error = 0
    for x, real_y, fake_y in zip(xs, real_ys, fake_ys):

        x = torch.Tensor(x).unsqueeze(1).to(device)
        real_y = torch.Tensor(real_y).unsqueeze(1).to(device)

        real_logit = D(real_y, x)
        real_label = torch.ones_like(real_logit)
        real_error = bce_loss(real_logit, real_label)
        d_real_error += real_error

        fake_logit = D(fake_y, x)
        fake_label = torch.zeros_like(fake_logit)
        fake_error = bce_loss(fake_logit, fake_label)
        d_fake_error += fake_error

    d_real_error = d_real_error / sample_size
    d_fake_error = d_fake_error / sample_size
    d_loss = d_real_error + d_fake_error
    d_loss.backward()
    d_opt.step()

    d_real_loss.append(d_real_error.cpu().detach().numpy())
    d_fake_loss.append(d_fake_error.cpu().detach().numpy())

    if unroll_steps:
        # * Unroll D
        d_backup = D.state_dict()
        for k in range(unroll_steps):
            # * Train D
            d_opt.zero_grad()
            d_real_error = 0
            d_fake_error = 0
            for x, real_y, fake_y in zip(xs, real_ys, fake_ys):

                x = torch.Tensor(x).unsqueeze(1).to(device)
                real_y = torch.Tensor(real_y).unsqueeze(1).to(device)

                real_logit = D(real_y, x)
                real_label = torch.ones_like(real_logit)
                real_error = bce_loss(real_logit, real_label)
                d_real_error += real_error

                fake_logit = D(fake_y, x)
                fake_label = torch.zeros_like(fake_logit)
                fake_error = bce_loss(fake_logit, fake_label)
                d_fake_error += fake_error

            d_real_error = d_real_error / sample_size
            d_fake_error = d_fake_error / sample_size
            d_loss = d_real_error + d_fake_error
            d_loss.backward()
            d_opt.step()

    # * Train G
    g_opt.zero_grad()
    g_error = 0
    for x in xs:
        x = torch.Tensor(x).unsqueeze(1).to(device)
        noise = torch.Tensor(sample_noise(1, noise_size)
                             ).unsqueeze(1).to(device)
        gen_y = G(x, noise)

        gen_logit = D(gen_y, x)
        gen_lable = torch.ones_like(gen_logit)
        gen_error = bce_loss(gen_logit, gen_lable)
        g_error += gen_error

    g_error = g_error / sample_size
    g_error.backward()
    g_opt.step()

    if unroll_steps:
        D.load_state_dict(d_backup)

    g_loss.append(g_error.cpu().detach().numpy())

    if epoch % log_interval == 0:
        torch.save(G.state_dict(), f'saved_models/epoch_{epoch}.pt')


fig = plt.figure(dpi=100)

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(range(len(d_real_loss)), d_real_loss, label='d real loss')
ax1.plot(range(len(d_fake_loss)), d_fake_loss, label='d fake loss')
ax1.plot(range(len(g_loss)), g_loss, label='g loss')
ax1.legend()

plt.savefig('hist.png')
plt.close()
