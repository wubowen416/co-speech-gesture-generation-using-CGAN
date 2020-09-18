from generating import generate_motion
from torch_dataset import TestSet
from models import Generator
from utils import load_model

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

os.makedirs('evaluation', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-r', help='whether calculate for real 1 or 0',
                    default='0')
parser.add_argument('-mp', help='Which parameters to import',
                    default='pre_trained_model/unroll_1000.pt')
args = parser.parse_args()


test_set = TestSet()

if args.r == '1':

    means = []
    ses = []

    real_frames = np.concatenate(test_set.Y_ori, axis=0)

    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=3)
    grid.fit(real_frames)

    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    scores = grid.best_estimator_.score_samples(real_frames)

    means.append(np.mean(scores))
    ses.append(np.std(scores)/np.sqrt(len(test_set)))

    df = pd.DataFrame([*zip(means, ses)], columns=['mean', 'se'])
    df.to_csv(f'evaluation/real.csv')

else:

    means = []
    ses = []

    for _ in range(3):

        G = load_model(args.mp)

        gened_seqs = []
        for x, y in test_set:
            outs = generate_motion(G, x, noise=None)
            gened_seqs.append(outs)
        gened_frames = np.concatenate(gened_seqs, axis=0)
        real_frames = np.concatenate(test_set.Y_ori, axis=0)

        params = {'bandwidth': np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params, cv=3)
        grid.fit(gened_frames)

        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

        scores = grid.best_estimator_.score_samples(real_frames)

        means.append(np.mean(scores))
        ses.append(np.std(scores)/np.sqrt(len(test_set)))

    df = pd.DataFrame([*zip(means, ses)], columns=['mean', 'se'])
    df.to_csv(f'evaluation/{os.path.basename(args.mp)}.csv')
