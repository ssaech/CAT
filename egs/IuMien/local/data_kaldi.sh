#!/bin/bash
set -euo pipefail

# Default to the local corpus location on this machine, but allow overrides.
audio_dir="${MIGHTLJSPEECH_WAV_DIR:-/Users/web/MightLJSpeech/MightLJSpeech-1.1/wavs}"
text_root="${MIGHTLJSPEECH_TEXT_ROOT:-/Users/web/MightLJSpeech/MightLJSpeech-1.1}"
output_dir="./data/src"
datasets=("train" "dev" "test")

if [ ! -d "$audio_dir" ]; then
    echo "Error: audio directory not found: $audio_dir"
    echo "Set MIGHTLJSPEECH_WAV_DIR to the folder that contains the .wav files."
    exit 1
fi

if [ -z "${KALDI_ROOT:-}" ]; then
    echo "Error: KALDI_ROOT is not specified."
    echo "Example: export KALDI_ROOT=/path/to/kaldi"
    exit 1
fi

if [ ! -d "$KALDI_ROOT" ] || [ ! -d "$KALDI_ROOT/egs/wsj/s5" ]; then
    echo "Error: KALDI_ROOT does not look like a valid Kaldi checkout: $KALDI_ROOT"
    exit 1
fi

for dataset_name in "${datasets[@]}"; do
    split_file="$text_root/${dataset_name}_data.txt"
    if [ ! -f "$split_file" ]; then
        echo "Error: split file not found: $split_file"
        echo "Expected train_data.txt, dev_data.txt, and test_data.txt under $text_root"
        exit 1
    fi

    mkdir -p "$output_dir/$dataset_name"
    : > "$output_dir/$dataset_name/wav.scp"
    : > "$output_dir/$dataset_name/text"

    while IFS=$'\t' read -r filename text; do
        audio_path="$audio_dir/$filename.wav"
        if [ ! -f "$audio_path" ]; then
            echo "Warning: missing audio file: $audio_path"
            continue
        fi
        echo -e "$filename\t$audio_path" >> "$output_dir/$dataset_name/wav.scp"
        echo -e "$filename\t$text" >> "$output_dir/$dataset_name/text"
    done < "$split_file"

    if [ ! -s "$output_dir/$dataset_name/text" ]; then
        echo "Error: no examples were written for $dataset_name"
        exit 1
    fi

    echo "wav.scp and text files generated successfully in $output_dir/$dataset_name."
done

data_dir=./data/src

for ti in dev train test; do
    awk '{print $1,$1}' "$data_dir/${ti}/text" > "$data_dir/${ti}/utt2spk"
    cp "$data_dir/${ti}/utt2spk" "$data_dir/${ti}/spk2utt"

    mv "$data_dir/$ti/wav.scp" "$data_dir/$ti/wav_mp3.scp"
    awk '{print $1 "\tffmpeg -i " $2 " -f wav -ar 16000 -ab 16 -ac 1 - |"}' \
        "$data_dir/$ti/wav_mp3.scp" > "$data_dir/$ti/wav.scp"
done

bash utils/data/data_prep_kaldi.sh \
    data/src/{train,dev,test} \
    --feat-dir=data/fbank \
    --nj=16 \
    --not-apply-cmvn
    
