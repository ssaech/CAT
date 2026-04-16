# Copyright 2023 Tsinghua SPMI Lab, Author: DongLukuan (330293721@qq.com)

import argparse
import os


def read_file(file_path, need_split=False):
    file_text = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if need_split:
                parts = line.split('\t', maxsplit=1)
                if len(parts) < 2:
                    continue
                file_text.append(parts[1])
            else:
                file_text.append(line)
    return file_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("MIGHTLJSPEECH_TEXT_ROOT", "/Users/web/MightLJSpeech/MightLJSpeech-1.1"),
        help="Directory containing train_data.txt, dev_data.txt and test_data.txt.",
    )
    args = parser.parse_args()

    train_data_path = os.path.join(args.data_root, "train_data.txt")
    dev_data_path = os.path.join(args.data_root, "dev_data.txt")
    test_data_path = os.path.join(args.data_root, "test_data.txt")

    train_data = read_file(train_data_path, True)
    dev_data = read_file(dev_data_path)
    test_data = read_file(test_data_path)

    new_test_data = []
    for line in test_data:
        parts = line.strip().split('\t', maxsplit=1)
        if len(parts) < 2:
            continue
        ids, sentence = parts
        if sentence in train_data:
            continue
        new_test_data.append(line)

    with open(test_data_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_test_data))

    new_dev_data = []
    for line in dev_data:
        parts = line.strip().split('\t', maxsplit=1)
        if len(parts) < 2:
            continue
        ids, sentence = parts
        if sentence in train_data:
            continue
        new_dev_data.append(line)

    with open(dev_data_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_dev_data))


if __name__ == "__main__":
    main()
