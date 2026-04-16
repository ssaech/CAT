# Copyright 2023 Tsinghua SPMI Lab, Author: DongLukuan (330293721@qq.com)

import random
import argparse
import os

def main_work(args):
    data_path = args.data_path
    if args.data_output_path is None:
        output_path = os.path.dirname(data_path)
    else:
        output_path = args.data_output_path
    with open(data_path, 'r') as input_file:
        data = input_file.readlines()

    data = [line.replace('|', '\t').replace('.wav','') for line in data]

    random.seed(args.seed)
    random.shuffle(data)

    num_parts = 10
    chunk_size = len(data) // num_parts
    chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_parts)]

    remainder = len(data) % num_parts
    for i in range(remainder):
        chunks[i].append(data[num_parts * chunk_size + i])

    select_train_data = [0, 1, 2, 3, 4, 5, 6, 7]

    os.makedirs(output_path, exist_ok=True)

    with open(f'{output_path}/train_data.txt', 'w', encoding='utf-8') as file:
        for i in select_train_data:
            file.writelines(chunks[i])
    with open(f'{output_path}/dev_data.txt', 'w', encoding='utf-8') as file:
        file.writelines(chunks[8])
    with open(f'{output_path}/test_data.txt', 'w', encoding='utf-8') as file:
        file.writelines(chunks[9])


if __name__ == "__main__":
        # 创建解析器对象
    parser = argparse.ArgumentParser(description="split data")
    
    # 添加位置参数
    parser.add_argument("--data_path", type=str, help="MightLJSpeech data path")
    parser.add_argument("--data_output_path", type=str, help="split data output path", default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the train/dev/test split.")
    args = parser.parse_args()
    main_work(args)
    
