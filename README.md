# co-speech-gesture-generation-using-generative-model

packages: pandas, tqdm, pydub, parselmouth, numpy, pyquaternion, pytorch


Instructions:

1. Download dataset that has two folder: speech and motion
2. Put speech and motion folder into dataset directory which shold be at root path
3. python data_processing/prepare_data.py data


    4. Use pre-trained: Using pytorch to load parameters provided.
        1. Scale input to (-1, 1) using train set
        2. Reshape to (T, N, dim)
        3. Reshape output from (T, N, dim) to (T, dim)
        4. Rescale back to 3D


    4. Train again(GPU required):
        1. python data_processing/create_vector.py data
