import jieba
import numpy as np
import os


threshold = 0.7

root = os.getcwd()
p_data = root + '\\data'
p_train = p_data + '\\train.tsv'
p_neg = root + '\\neg.txt'

def get_word_vector(s1, s2):
    """
    input: 两个句子\n
    output: 两个句子的向量\n
    function: 输入两个句子，给出numpy数组的句子中词的词频表示
    """

    cut1 = jieba.cut(s1)
    cut2 = jieba.cut(s2)

    list_word1 = (','.join(cut1)).split(',')
    list_word2 = (','.join(cut2)).split(',')
    print(list_word1)
    print(list_word2)

    key_word = list(set(list_word1 + list_word2))#取并集
    print(key_word)

    word_vector1 = np.zeros(len(key_word))#给定形状和类型的用0填充的矩阵存储向量
    word_vector2 = np.zeros(len(key_word))

    for i in range(len(key_word)):#依次确定向量的每个位置的值
        for j in range(len(list_word1)):#遍历key_word中每个词在句子中的出现次数
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    print(word_vector1)#输出向量
    print(word_vector2)
    return word_vector1, word_vector2

def cosine(s1, s2):
    """
    input: 两个句子\n
    output: 两个句子的余弦相似度 float\n
    function: 输入两个句子，给出余弦相似度
    """
    v1, v2 = get_word_vector(s1, s2)
    return float(np.sum(v1 * v2))/(np.linalg.norm(v1) * np.linalg.norm(v2))

def data_handler(p_train):
    """
    input: 训练数据的路径\n
    output: 问题，回答，label的三个列表\n
    function: 读入并处理训练数据文件
    """
    qus = []    # 存放问题
    ans = []    # 存放回答
    label = []  # 存放label
    with open(file=p_train, mode='r', encoding='utf-8') as f:
        raw = f.readlines()[:-1]
        for r in raw:
            temp = r.strip().split('\t')
            qus.append(temp[0])
            ans.append(temp[1])
            label.append(int(temp[2]))
    return qus, ans, label

def negative_generator(p_neg, qus, ans, label):
    neg = 
    pass



if __name__ == '__main__':
    qus, ans, label = data_handler(p_train)
    pass

