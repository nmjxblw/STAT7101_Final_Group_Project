"""莎士比亚数据集生成器模块"""

from concurrent.futures import thread
from datetime import datetime
import os
import random
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import re
from dotenv import load_dotenv
from pathlib import Path
import sys

load_dotenv()

FILE_PATH: Path = Path.joinpath(
    Path.cwd(), os.getenv("FILE_NAME", "t8.shakespeare.txt")
)
"""原文本文件路径"""
TRAIN_FILE_PATH: Path = Path.joinpath(Path.cwd(), "shakespeare_train_set.txt")
TEST_FILE_PATH: Path = Path.joinpath(Path.cwd(), "shakespeare_test_set.txt")
SPLIT_STRATEGY: int = int(os.getenv("SPLIT_STRATEGY", "0"))
# TODO: 根据随机行数生成测试数据集，将剩余部分作为训练集
random_factor: int = datetime.now().microsecond % 1000
""" 随机因子，用于生成随机行数 """
random.seed(random_factor)

with open(FILE_PATH, "rb") as file:
    file_content: str = file.read().decode("utf-8")


processed_content: str = file_content
processed_chunks: list[str] = []
if SPLIT_STRATEGY == 0:

    processed_content = re.sub(r"[^\sA-Za-z.]", " ", file_content).lower()

    processed_content = re.sub(r"[\s\n]+", " ", processed_content).strip()

    processed_chunks = re.split(
        r"[.]", processed_content
    )  # 将文本按句号和逗号分割为多个片段
else:
    processed_content = re.sub(r"[^\sA-Za-z]", " ", file_content).lower()

    processed_content = re.sub(r"[\s]+", " ", processed_content).strip()

    processed_chunks = re.split(r"\s", processed_content)
# TODO: 随机选择1/10的文本作为测试集
test_count: int = len(processed_chunks) // 10

test_indices = sorted(random.sample(range(len(processed_chunks)), test_count))

_thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)
futures: list[Future] = []
for chunk in processed_chunks:
    futures.append(
        _thread_pool.submit(
            lambda chunk: chunk.strip() + "\n",
            chunk,
        )
    )
with open(TEST_FILE_PATH, "w", encoding="utf-8") as test_file:
    with open(TRAIN_FILE_PATH, "w", encoding="utf-8") as train_file:
        for i, future in enumerate(as_completed(futures)):
            line = future.result()
            if i in test_indices:
                test_file.write(line)
            else:
                train_file.write(line)

_thread_pool.shutdown(wait=True)
