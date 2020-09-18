
wav_path='data/test/inputs/audio1094.wav'

mkdir results
mkdir results/gened

for p in './saved_models'/*
do

    model_name="$(basename -- $p)"

    echo "$model_name"

    npy_path='results/gened/npy_tmp.npy'

    python generating.py $wav_path $npy_path -mp $p

    mute_mp4_path='./results/gened/mp4_tmp.mp4'

    python example_scripts/make_mute_mp4.py $npy_path $mute_mp4_path

    mp4_path='results/gened/video_'$model_name'.mp4'

    ffmpeg -i $mute_mp4_path -i $wav_path -c:v copy -map 0:v:0 -map 1:a:0 $mp4_path

    rm $npy_path
    rm $mute_mp4_path

done