import argparse
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
sns.set()


def make_mute_mp4_single(seq, mp4_filepath):

    print("Making anime of length {}...".format(seq.shape[0]))
    n_frames = seq.shape[0]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [2, 5], [5, 6], [6, 7],
        [2, 8], [8, 9], [9, 10]
    ]

    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200)

    def update(i):
        label = f'timestep {i}'

        x, y = seq[i, ::3], seq[i, 1::3]
        ax.cla()
        ax.set_xlim(-40, 40)
        ax.set_ylim(40, 110)
        ax.axis('off')
        for j in lines:
            ax.plot(x[j], y[j])
        ax.scatter(x, y, s=10)
        return fig, ax

    anim = FuncAnimation(fig, update, frames=range(n_frames), interval=50)
    anim.save(mp4_filepath, writer='ffmpeg')
    print("mute mp4 done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('npy_path', help='Path of .npy motion file')
    parser.add_argument(
        'mp4_path', help='Path for saving .mp4 file')
    args = parser.parse_args()

    motion = np.load(args.npy_path)
    make_mute_mp4_single(motion, args.mp4_path)
