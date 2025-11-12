import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier



def preprocess(content):
    letters = [i for i in 'abcdefghijklmnopqrstuvwxyz']
    #文本预处理：全小写，移除标点，空格分隔
    text = ''
    for i in content.lower():
        if i in letters:
            text += i
        elif text[-1] != ' ':
            text += ' '
    return text




class ImprovedLetterFrequencyDecisionTree:
    def __init__(self, max_depth=5,verbose=False):
        self.max_depth = max_depth
        self.model = None
        self.verbose=verbose
    
    def extract_features(self, wordlist, letter):
        features=[]
        """从单词表中提取某个字母的特征"""
        if not wordlist:
            return [0, 0, 0, 0]
            
        text = ' '.join(wordlist).lower()
        total_letters = len([c for c in text if c.isalpha()])
        
        # 特征计算
        # 特征1
        letter_count = text.count(letter)
        global_freq = letter_count / total_letters if total_letters > 0 else 0
        features.append(global_freq)
        
        # 特征2
        starts_with_count = sum(1 for word in wordlist if word.startswith(letter))
        start_freq = starts_with_count / len(wordlist) if wordlist else 0
        features.append(start_freq)

        # 特征3
        ends_with_count = sum(1 for word in wordlist if word.endswith(letter))
        end_freq = ends_with_count / len(wordlist) if wordlist else 0
        features.append(end_freq)

        # 特征4
        # 位置分布特征
        first_half_count = 0
        for word in wordlist:
            half_point = max(1, len(word) // 2)
            first_half_count += word[:half_point].count(letter)
        
        first_half_ratio = first_half_count / letter_count if letter_count > 0 else 0
        features.append(first_half_ratio)

        return features
    
    def generate_realistic_training_data(self, base_wordlist, num_variations=100):
        """
        生成更真实的训练数据：通过文本变换模拟不同文本特征
        """
        X = []  # 特征
        y = []  # 标签
        
        # 1. 原始文本特征（暂不使用）
        # self._add_text_variation(X, y, base_wordlist, "original")
        
        # 2. 不同长度的文本片段（模拟不同规模的加密文本）
        for i in range(num_variations):
            # 随机采样部分文本
            sample_size = max(10, min(len(base_wordlist), 
                                    random.randint(len(base_wordlist)//10, len(base_wordlist))))
            sampled_words = random.sample(base_wordlist, sample_size) if len(base_wordlist) > sample_size else base_wordlist
            
            self._add_text_variation(X, y, sampled_words, f"sample_{i}")
        
        return np.array(X), np.array(y)
    
    def _add_text_variation(self, X, y, wordlist, variation_name):
        """为特定文本变体添加训练数据"""
        # 为每个字母提取特征
        for label in 'abcdefghijklmnopqrstuvwxyz':
            features = self.extract_features(wordlist, label)
            X.append(features)
            y.append(label)
            if self.verbose:
                print(f"features:{features},label:{label}")
    
    def train_with_real_data(self, training_texts):
        """
        使用真实的不同文本来训练，获得更鲁棒的特征
        training_texts: 不同主题/风格的文本列表
        """
        X_all = []
        y_all = []
        
        for text in training_texts:
            # 预处理文本
            words = preprocess(text).split()
            if len(words) < 10:  # 跳过太短的文本
                continue
                
            X, y = self.generate_realistic_training_data(words, num_variations=5)
            X_all.extend(X)
            y_all.extend(y)
        
        # 训练模型
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=42,
            min_samples_split=20,
            min_samples_leaf=10
        )
        self.model.fit(X_all, y_all)
        
        print(f"训练完成，样本数量: {len(X_all)}")
    
    def train_with_base_text(self, base_wordlist,num_variations):
        """使用基础文本训练（简化版，适用于单一文本）"""
        X, y = self.generate_realistic_training_data(base_wordlist, num_variations)
        
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=42
        )
        self.model.fit(X, y)
        print(f"训练完成，样本数量: {len(X)}")
    
    def predict_permutation(self, encrypted_wordlist):
        """预测整个排列映射，返回 {明文字母: 密文字母} 格式"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        letter_probabilities = {}
        plain_letters='abcdefghijklmnopqrstuvwxyz'
        for cipher_letter in plain_letters:
            features = self.extract_features(encrypted_wordlist, cipher_letter)
            proba = self.model.predict_proba([features])[0]
            letter_probabilities[cipher_letter] = proba
            if self.verbose:
                    # 获取前3个最有可能的明文字母
                top_indices = np.argsort(proba)[-3:][::-1]  # 从高到低
                print(f"密文 '{cipher_letter}' 最可能解密为:")
                for idx in top_indices:
                    plain_letter = plain_letters[idx]
                    probability = proba[idx] * 100
                    print(f"  {plain_letter}: {probability:.1f}%")
                print("-" * 30)
        
        
        # 使用匈牙利算法解决冲突
        return self._resolve_conflicts_with_hungarian(letter_probabilities)
    
    def _resolve_conflicts_with_hungarian(self, probabilities):
        """使用匈牙利算法找到最优字母映射，返回 {明文字母: 密文字母}"""
        from scipy.optimize import linear_sum_assignment
        
        letters = 'abcdefghijklmnopqrstuvwxyz'
        cost_matrix = np.zeros((26, 26))
        
        for i, cipher_letter in enumerate(letters):
            for j, plain_letter in enumerate(letters):
                prob = probabilities[cipher_letter][j]
                cost_matrix[i, j] = -np.log(prob + 1e-8)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mapping = {}
        for i, j in zip(row_ind, col_ind):
            # 修改这一行：将映射方向反转
            # 原来是 mapping[密文字母] = 明文字母
            # 现在改为 mapping[明文字母] = 密文字母
            mapping[letters[j]] = letters[i]  # 注意i和j的位置交换了！

        sorted_mapping = {k: mapping[k] for k in sorted(mapping.keys())}
        return sorted_mapping
        


def calculate_accuracy(y_true, y_pred):
    if set(y_true.keys()) != set(y_pred.keys()):
        raise ValueError("实际映射和预测映射的键必须相同")
    correct_count = 0
    for letter in y_true:
        if y_pred[letter] == y_true[letter]:
            correct_count += 1
    return correct_count / len(y_true)


file_path = r"C:\Users\Lenovo\Desktop\7101\text.txt"
with open(file_path, mode="r", encoding="utf-8") as f:
    plaintext = f.read()
plaintext = preprocess(plaintext)
print(len(plaintext))
    
base_wordlist = plaintext.lower().split()
# 初始化模型
model = ImprovedLetterFrequencyDecisionTree(max_depth=6,verbose=False)
model.train_with_base_text(base_wordlist,100)



print("随机加密测试效果")
test_mapping = {'a': 'x', 'b': 'y', 'c': 'z', 'd': 'a', 'e': 'b', 
                'f': 'c', 'g': 'd', 'h': 'e', 'i': 'f', 'j': 'g',
                'k': 'h', 'l': 'i', 'm': 'j', 'n': 'k', 'o': 'l',
                'p': 'm', 'q': 'n', 'r': 'o', 's': 'p', 't': 'q',
                'u': 'r', 'v': 's', 'w': 't', 'x': 'u', 'y': 'v', 'z': 'w'}
encrypted_words = []
for word in base_wordlist:
    encrypted_word = ''.join(test_mapping.get(c, c) for c in word)
    encrypted_words.append(encrypted_word)

# 预测排列
predicted_mapping = model.predict_permutation(encrypted_words)
print("实际映射（y_label）:", test_mapping)
print("预测映射（y_pred ）:", predicted_mapping)
print("预测准确率:",calculate_accuracy(test_mapping,predicted_mapping))



print("当前方法思想：决策树，根据单个字母的特征为输入（如字母在样本的频率p(letter)等特征）,预测的标签作为输出")
print("---模型训练---")
print("当前代码运行根据单一原始文本(text.txt)进行一百次操作。")
print("每次操作随机取原始单词量[0.1，1]概率的单词（无序且随机，可跳词取）集合S")
print("根据集合S构建原始字母的统计特征，如出现频率等")
print("根据100次集合S的采样得到的26*100个X-y 样本-标签数据进行训练 （现在的X为四维，即四种特征）即可得到决策树")

print("---模型预测---")
print("先生成加密的密钥，将原文本加密，并得到每个加密后字母的X （即统计特征）")
print("根据统计特征X让模型预测y_pred")
print("对比y和y_pred")
