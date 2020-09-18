from prosodic_feature import extract_prosodic_features
from bvh_to_3d import vectorize_bvh

import sys
import pandas as pd
import tqdm
import pickle


def shorten(arr1, arr2):
    min_len = min(len(arr1), len(arr2))

    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]

    return arr1, arr2


def create_vectors(audio_filename, gesture_filename):
    """
    Extract features from a given pair of audio and motion files
    Args:
        audio_filename:    file name for an audio file (.wav)
        gesture_filename:  file name for a motion file (.bvh)
        nodes:             an array of markers for the motion

    Returns:
        input_with_context   : speech features
        output_with_context  : motion features
    """

    # Step 1: speech features

    input_vectors = extract_prosodic_features(audio_filename)

    # Step 2: Vectorize BVH

    output_vectors = vectorize_bvh(gesture_filename)

    # Step 3: Align vector length
    input_vectors, output_vectors = shorten(input_vectors, output_vectors)

    return input_vectors, output_vectors


def create(name):
    """
    Create a dataset
    Args:
        name:  dataset: 'train' or 'test' or 'dev
        nodes: markers used in motion caption

    Returns:
        nothing: saves numpy arrays of the features and labels as .npy files

    """
    DATA_FILE = pd.read_csv(DATA_DIR + '/gg-' + str(name) + '.csv')
    X, Y = [], []

    for i in tqdm.tqdm(range(len(DATA_FILE))):
        input_vectors, output_vectors = create_vectors(
            DATA_FILE['wav_filename'][i], DATA_FILE['bvh_filename'][i])

        X.append(input_vectors)
        Y.append(output_vectors)

    x_file_name = DATA_DIR + '/X_' + str(name) + '.p'
    y_file_name = DATA_DIR + '/Y_' + str(name) + '.p'
    with open(x_file_name, 'wb+') as f:
        pickle.dump(X, f)
    with open(y_file_name, 'wb+') as f:
        pickle.dump(Y, f)


if __name__ == "__main__":

    # Specify data dir
    DATA_DIR = sys.argv[1]

    # create('train')
    create('test')
