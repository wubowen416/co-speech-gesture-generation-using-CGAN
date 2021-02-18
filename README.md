# co-speech-gesture-generation-using-CGAN
https://doi.org/10.3390/electronics10030228

Required packages: pandas, tqdm, pydub, parselmouth, numpy, pyquaternion, pytorch

ffmpeg is required for  `example_scripts/gen_all_models.sh`

## Directly use pre-trained model to generate motions.

The model we selected is stored as `pre_trained_model/unroll_1000.pt`. Parameters are compatible with class `Generator`  in `models.py`.

1. Clone this repository
2. Download a dataset from https://www.dropbox.com/sh/j419kp4m8hkt9nd/AAC_pIcS1b_WFBqUp5ofBG1Ia?dl=0
3. Create a directory named `dataset `and put two directories `motion/` and `speech/` under `dataset/`
4. `python data_processing/prepare_data.py data`  to split dataset to train, dev, and test set (as we used data in train set to scale input and output). Then run `python data_processing/create_vector.py`
5.  Play with `generating.py`. Arguments of this function are: (1)wav file path, (2)output save path, (3)(optional)model parameters path. This function will process wav file and produce motion using parameters provided. You can specify the noise vector used to generate motions as one argument for function `generate_motion`. The result will be .npy format. E.g.,
```
python generating.py data/test/inputs/audio1094.wav test_output.npy -mp pre_trained_model/unroll_1000.pt
```
6.  If you want to see video, use `example_scripts/make_mute_mp4.py`. The first argument is to specify .npy file, the second argument is to set save path. However, the video is without audio.E.g., 
```
python example_scripts/make_mute_mp4.py test_output.npy test_video.mp4
```
7. Attach audio to the mute video(ffmpeg required). Or you can find your own way.
```
ffmpeg -i test_video.mp4 -i data/test/inputs/audio1094.wav -c:v copy -map 0:v:0 -map 1:a:0 test_video.mp4
```

## Train your own model(GPU only)
Notice GAN is not guaranteed to produce same result for every training, so your result could be different from ours.

1. Following 1-4 in **Directly use pre-trained model to generate motions**.
2. `python train.py` to train a new model. The architecture of model is defined in `models.py`. Hyper-parameters are defined in `train.py`. Params of generator are periodically saved during training.
3. You may want to check your result. `example_scripts/gen_all_models.sh` is a function for generating videos with audio for all saved models for one audio input.






