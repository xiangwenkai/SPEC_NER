import fsspec.config
import pandas as pd
import re
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from separate_spec import sentence_split, insertSeparator
from utils import get_ents_rep, sample_preprocess

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sample_preprocess(s):
    # 对训练和预测样本进行预处理
    # 分号后面加空格
    if s == '':
        return s
    s = re.sub(';', '; ', s)
    s = s.replace('||', ' || ')
    s = s.replace('(', ' (')
    s = s.replace(')', ') ')
    # 多个空格变成一个
    s = re.sub('[ ]+', ' ', s)
    return s


def get_ents_rep(raws_list, ner_predict_list):
    '''
    实体用特殊字符替换
    :param raws_list:
    :param ner_predict_list: example:[('a+b',3,6,'FORMULA')]
    :return:
    '''
    ss = []
    for raw, entities in zip(raws_list, ner_predict_list):
        if type(entities) == str:
            entities = eval(entities)
        if len(entities) == 0:
            ss.append(raw)
            continue
        inds, cats = [0], []
        for ent in entities:
            inds.extend([ent[1], ent[2]])
            cats.append(ent[3])
        inds.append(-1)
        ent_map = {'FUNC': '[function]', 'FORMULA': '[formula]', 'FORMAT': '[format]', 'VAR': '[variable]',
                   'QUOT': '[quotation]'}
        cats = [ent_map[x] for x in cats]
        m = len(cats)
        k = 0
        s = ''
        for i in range(0, len(inds), 2):
            if k <= m - 1:
                s += raw[inds[i]:inds[i + 1]] + cats[k]
            k += 1
        ss.append(s)
    return ss


def get_sentence_embedding(sentence_list):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    embeddings = []
    for s in sentence_list:
        encoded_input = tokenizer(s, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            emb = list(sentence_embeddings.detach().cpu().numpy()[0])
        embeddings.append(emb)
    return embeddings


def get_cluster(embedding_list, k=10):
    '''
    使用Kmeans预测句子聚类类别
    :param embedding_list:
    :param k:
    :return:
    '''
    clu = KMeans(n_clusters=k)
    clu.fit(embedding_list)
    return clu.labels_


def kmeans_inertia(embedding_list, k_min=10, k_max=30, graph_savedir='AZ/images/inertia.png'):
    SSE = []
    for k in range(k_min, k_max):
        clu = KMeans(n_clusters=k)
        clu.fit(embedding_list)
        SSE.append(clu.inertia_)
    # plt.style("ggplot")
    plt.plot(range(k_min, k_max), SSE, 'o-')
    plt.savefig(graph_savedir, dpi=200)


def ner_predict(model_path, spec_list):
    nlp = spacy.load(model_path)
    pre_spec_list = []
    for x in spec_list:
        pre_spec_list.append([(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in nlp(x).ents])
    return pre_spec_list


def specPatternAnalysis(file_path, model_path='', spec_col='', k=10):
    """
    file: .xlsx
    targetDomain: "AE"
    """
    #  读入spec excel 文件
    df_all = pd.read_excel(file_path)

    df_all = df_all[df_all[spec_col].notnull()]

    spec_list = df_all[spec_col].tolist()
    spec_list = [sample_preprocess(x) for x in spec_list]

    # 预测spec
    ner_predict_list = ner_predict(model_path=model_path, spec_list=spec_list)

    # 实体替换
    df_all['rep'] = get_ents_rep(spec_list, ner_predict_list)

    df_all['rep'] = df_all['rep'].apply(sentence_split)
    df_all['rep'] = df_all['rep'].apply(insertSeparator)
    df_all['sentence'] = df_all['rep'].apply(lambda x: x.split('@@@@'))
    # 去除空值
    df_all['sentence'] = df_all['sentence'].apply(lambda x: [i for i in x if i != ''])

    df_all = df_all.explode('sentence')

    embedding_list = get_sentence_embedding(df_all['sentence'])

    clusters = get_cluster(embedding_list, k=k)

    df_all['cluster'] = clusters
    return df_all

'''
# 读取数据
data = pd.read_excel('AZ/data/AZ_Corporate_spec_predict.xlsx')
print(data.columns)
# 原始spec
raws = data['Map Definition']
# ner预测值
ner_predict_list = data['Map Mode Predicted']

# 实体替换
data['rep'] = get_ents_rep(raws, ner_predict_list)

# 分句
data['sentence'] = data['rep'].apply(lambda x: sentence_split(x).split('@@@@'))
data = data.explode('sentence')
data = data[data['sentence'] != '']

# 获取句子embeddings
embeddings = get_sentence_embedding(data['sentence'].tolist())

# 获取聚类结果
clusters = get_cluster(embedding_list=embeddings, k=2)
data['cluster'] = clusters

# 保存聚类结果
data.to_excel('AZ/data/AZ_Corporate_spec_cluster.xlsx')
'''
