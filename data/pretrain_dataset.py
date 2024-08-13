import os
import random
import re
import torch
from torch.utils.data import Dataset
from data.utils import pre_caption
import glob
import numpy as np
import concurrent.futures
from datasets import load_from_disk

class GigaSpeech(Dataset):
    def __init__(self, dataset_root_dir,feat_dir_root, max_words=50):
        """
        参数:
        feat_dir_root (str): 包含所有特征文件的根目录。
        dataset_root_dir (str): 包含所有子目录的数据集根目录。
        max_words (int): 文本描述的最大单词数。
        """
        self.feat_dirs = [os.path.join(feat_dir_root, d) for d in os.listdir(feat_dir_root) if os.path.isdir(os.path.join(feat_dir_root, d))]
        self.max_words = max_words

        # 加载所有子目录中的数据集
        self.datasets = self.load_all_datasets(dataset_root_dir)

        # 计算总长度
        self.total_len = sum(len(dataset) for dataset in self.datasets)
        self.dataset_boundaries = self._compute_dataset_boundaries()

    def load_all_datasets(self, root_dir):

        datasets = []

        # 遍历根目录下的所有子目录
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)

            # 检查路径是否为目录
            if os.path.isdir(subdir_path):
                # 尝试从子目录加载数据集
                try:
                    dataset = load_from_disk(subdir_path)
                    datasets.append(dataset)
                except Exception as e:
                    print(f"Error loading dataset from {subdir_path}: {e}")

        return datasets

    def _compute_dataset_boundaries(self):

        boundaries = []
        current_boundary = 0
        for dataset in self.datasets:
            current_boundary += len(dataset)
            boundaries.append(current_boundary)
        return boundaries

    def __len__(self):
        return self.total_len

    def _find_dataset_index(self, index):
        """
        给定全局索引，找到对应的子数据集和相对索引。
        """
        for i, boundary in enumerate(self.dataset_boundaries):
            if index < boundary:
                if i == 0:
                    return i, index
                else:
                    return i, index - self.dataset_boundaries[i - 1]
        raise IndexError("Index out of bounds")

    def clean(self, s):
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'"([^"]*)"', r'\1', s)
        s = re.sub(r"'([^']*)'", r'\1', s)
        return s

    def find_npz_file_in_dir(self, feat_dir, segment_id):
        npz_path = os.path.join(feat_dir, segment_id + '.npz')
        if os.path.exists(npz_path):
            return npz_path
        return None

    def find_npz_file(self, segment_id):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.find_npz_file_in_dir, feat_dir, segment_id) for feat_dir in self.feat_dirs]
            for future in concurrent.futures.as_completed(futures):
                npz_path = future.result()
                if npz_path is not None:
                    return npz_path
        raise FileNotFoundError(f"{segment_id}.npz not found in any of the provided directories.")

    def __getitem__(self, index):
        dataset_index, relative_index = self._find_dataset_index(index)
        dataset = self.datasets[dataset_index]

        tag = random.randint(1, 5)
        segment_id = dataset["segment_id"][relative_index]
        caption = dataset["text_description"+str(tag)][relative_index]
        caption = self.clean(caption)
        caption = pre_caption(caption, self.max_words)

        # Find the correct `.npz` file in the directories using optimized search
        npz_path = self.find_npz_file(segment_id)
        hubert_fea = torch.from_numpy(np.load(npz_path)['arr_0'])

        return segment_id, hubert_fea, caption


if __name__ == '__main__':
    # 根目录，包含所有子目录及其数据集
    dataset_root_dir = "****"

    # 根目录，包含所有 npz 文件的子目录
    feat_dir_root = "****"

    dataset = GigaSpeech(feat_dir_root, dataset_root_dir)
    print(len(dataset))
    print(dataset[4])
