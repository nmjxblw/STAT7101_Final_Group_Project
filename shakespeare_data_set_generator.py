"""莎士比亚数据集生成器模块"""

from concurrent.futures import thread
from datetime import datetime
import os
import random
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import re

import test
from dotenv import load_dotenv
from pathlib import Path
import sys


# region 变量定义与初始化
load_dotenv()

FILE_PATH: Path = Path.joinpath(
    Path.cwd(), os.getenv("FILE_NAME", "t8.shakespeare.txt")
)
"""原文本文件路径"""
TRAIN_FILE_PATH: Path = Path.joinpath(Path.cwd(), "shakespeare_train_set.txt")
"""训练集文件路径"""
TEST_FILE_PATH: Path = Path.joinpath(Path.cwd(), "shakespeare_test_set.txt")
"""测试集文件路径"""
VALIDATION_FILE_PATH: Path = Path.joinpath(Path.cwd(), "shakespeare_validation_set.txt")
"""验证集文件路径"""
SPLIT_STRATEGY: int = int(os.getenv("SPLIT_STRATEGY", "0"))
"""文本分割策略"""
SINGLE_VOCAB_SET_SIZE: int = int(os.getenv("SINGLE_VOCAB_SET_SIZE", "300"))
"""单个词汇表的大小"""
# 根据随机行数生成测试数据集，将剩余部分作为训练集
random_factor: int = datetime.now().microsecond % 1000
""" 随机因子，用于生成随机行数 """
random.seed(random_factor)
# endregion

with open(FILE_PATH, "rb") as file:
    file_content: str = file.read().decode("utf-8")


processed_content: str = file_content
""" 预处理后的文本内容 """
processed_chunks: list[str] = []
""" 预处理后的切割片段列表 """

# region 根据分割策略处理文本内容
if SPLIT_STRATEGY == 0:
    # 句子分割策略
    # 只保留空格、字母字符和句号，其他字符替换为空格
    processed_content = re.sub(r"[^\sA-Za-z.]", " ", file_content).lower()
    # 将多个连续的空白字符替换为单个空格
    processed_content = re.sub(r"[\s\n]+", " ", processed_content).strip()
    # 按句号分割文本
    processed_chunks = re.split(
        r"[.]", processed_content
    )  # 将文本按句号和逗号分割为多个片段
elif SPLIT_STRATEGY == 1:
    # 单词组分割策略
    # 只保留空格和字母字符，其他字符替换为空格
    processed_content = re.sub(r"[^\sA-Za-z]", " ", file_content).lower()
    # 将多个连续的空白字符替换为单个空格
    processed_content = re.sub(r"[\s]+", " ", processed_content).strip()
    # 按空格分割文本
    processed_chunks = re.split(r"\s", processed_content)
else:
    raise ValueError("不支持的分割策略")
# endregion

# 分别选择1/10的chunk作为测试集和验证集
test_count: int = len(processed_chunks) // 10
""" 测试集样本数量 """
validation_count: int = len(processed_chunks) // 10
""" 验证集样本数量 """
train_count: int = len(processed_chunks) - test_count - validation_count
""" 训练集样本数量 """
# 生成不重复的随机索引列表
train_indices: list | list[int] = [i for i in range(len(processed_chunks))]
test_indices: list | list[int] = sorted(random.sample(train_indices, test_count))
""" 测试集索引列表 """
train_indices = [i for i in train_indices if i not in test_indices]
validation_indices: list | list[int] = sorted(
    random.sample(train_indices, validation_count)
)
""" 验证集索引列表 """
train_indices = [i for i in train_indices if i not in validation_indices]

if SPLIT_STRATEGY == 1:
    # 对于单词组分割策略，按词汇表大小重新组合片段
    train_indices = [
        indices for indices in zip(*[iter(train_indices)] * SINGLE_VOCAB_SET_SIZE)
    ]
# 使用多线程处理文本片段的写入
_thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=(os.process_cpu_count() or 1) + 4
)
futures: list[Future] = []


def _single_sample_processor(index: int | list[int]) -> str:
    """处理单个样本片段的函数"""
    line: str = ""
    if isinstance(index, int):
        line = processed_chunks[index]
    else:
        line_set = []
        for idx in index:
            line_set.append(processed_chunks[idx])
        line = " ".join(line_set)
    return line


for index in test_indices:
    futures.append(
        _thread_pool.submit(
            _single_sample_processor,
            index,
        )
    )
with open(TEST_FILE_PATH, "w", encoding="utf-8") as test_file:
    with open(TRAIN_FILE_PATH, "w", encoding="utf-8") as train_file:
        with open(VALIDATION_FILE_PATH, "w", encoding="utf-8") as validation_file:
            for i, future in enumerate(as_completed(futures)):
                line = future.result()
                if i in test_indices:
                    test_file.write(line)
                elif i in validation_indices:
                    validation_file.write(line)
                else:
                    train_file.write(line)

_thread_pool.shutdown(wait=True)
