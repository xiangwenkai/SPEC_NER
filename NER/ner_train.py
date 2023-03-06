import time
import numpy as np
import pandas as pd
from random import choice
import re
import random
from spacy.tokens import DocBin
from tqdm import tqdm
import spacy
from ner_build_data import get_one_sample_unnested
from spacy import displacy
from spacy.language import Language
import re


def sample_preprocess(s):
    # 对训练和预测样本进行预处理
    # 分号后面加空格
    if s == '':
        return s
    s = re.sub(';', '; ', s)
    s = s.replace('||', ' || ')
    s = s.replace('(', ' (')
    s = s.replace(')', ') ')
    s = s.replace("”", '"')
    # s = re.sub('([A-Z]{1})\.[ ]{1}', lambda x: x.group(1) + ' . ', s)
    # 多个空格变成一个
    s = re.sub('[ ]+', ' ', s)
    return s



df_vars = pd.read_excel('entity_raw/df_vars.xlsx')
df_quots = pd.read_excel('entity_raw/df_quots.xlsx')
df_formats = pd.read_excel('entity_raw/df_formats.xlsx')
df_formulas = pd.read_excel('entity_raw/df_formulas.xlsx')
df_funcs = pd.read_excel('entity_raw/df_funcs.xlsx')
df_rep = pd.read_excel('entity_raw/res_sample.xlsx')


vars = df_vars['var'].tolist()
quots = df_quots['quot'].tolist()
formats = df_formats['format'].tolist()
formulas = df_formulas['formula'].tolist()
funcs = df_funcs['func'].tolist()
rep_samples = df_rep['res'].tolist()
rep_samples = [sample_preprocess(s) for s in rep_samples]


for split, num in [('train', 100000), ('val', 5000)]:
    m = num
    ner_data = [get_one_sample_unnested(rep_samples, vars, quots, formats, formulas, funcs) for _ in range(num)]
    # ========Convert the annotated data into the spaCy bin object=======
    nlp = spacy.blank('en')  # load a new spacy model
    db = DocBin()  # create a DocBin object
    for text, annot in tqdm(ner_data):  # data in previous format
        doc = nlp.make_doc(text)  # create doc object from text
        ents = []
        for start, end, label in annot['entities']:  # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode='strict')
            if span is None:
                # print(f'Skipping entity: {text[max(0,start-2): end+5]}')
                # print(f'Skipping entity')
                m -= 1
            else:
                ents.append(span)
        if len(ents) == len(annot['entities']):
            doc.ents = ents  # label the text with the ents
            db.add(doc)
            # try:
            #     doc.ents = ents  # label the text with the ents
            #     db.add(doc)
            # except:
            #     print("fail")
            #     m -= 1
    print(m)
    db.to_disk(f'./AZ/{split}.spacy')  # save the docbin object


# get complete cfg file
'''
python -m spacy init fill-config ./base_config_trf.cfg ./config_trf.cfg
'''
# 安装en_core_web_trf
'''
python -m spacy download en_core_web_trf
'''

# train command:
'''
# python -m spacy train config.cfg --output ./ --paths.train ./train.spacy --paths.dev ./val.spacy
# python -m spacy train config_sm.cfg --output ./ --paths.train ./train.spacy --paths.dev ./val.spacy
# python -m spacy train config_trf.cfg --output ./ --paths.train ./train.spacy --paths.dev ./val.spacy
# nohup python -m spacy train config_trf.cfg --output ./ --paths.train ./train.spacy --paths.dev ./val.spacy >train_ner_trf.log &
'''


'''
# simple inference
nlp = spacy.load('AZ/model-best-trf/') #load the model
sentence = get_one_sample_unnested(rep_samples)[0]
print(sentence)

sentence = '"ADY = ADT - ADSL.TRTSDT + 1 if ADT >= ADSL.TRTSDT; ADY = ADT - ADSL.TRTSDT if ADT < ADSL.TRTSDT."'
sentence = 'Set to LB.LBSTRESN. If LB.LBSTRESC = "<X" or ">X", then set to X.'
sentence = 'Set to DTHDTN - TRTDPEDT;if >=0 then DTHDLDY + 1.'
sentence = 'AED=1; if (DTHDTN+TRTDPEDT=1 and there is something wrong)'
doc = nlp(sentence)
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
displacy.serve(doc, style="dep")
'''



