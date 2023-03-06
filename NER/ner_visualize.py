import spacy
from spacy import displacy
import pandas as pd
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
    # 多个空格变成一个
    s = re.sub('[ ]+', ' ', s)
    return s


def ner_vis_divs(spec_list, model_path='AZ/model-best/'):
    ner = spacy.load(model_path)  # load the model
    docs = ner.pipe(spec_list)

    # colors = {'FUNC': "#FF6A6A", "FORMULA": "#EE82EE", "VAR": "#6495ED", "QUOT": "#9AFF9A","FORMAT":"#9C9C9C"}
    colors = {'FUNC': "#FF6A6A", "FORMULA": "#FF6A6A", "VAR": "#6495ED", "QUOT": "#9AFF9A", "FORMAT": "#9C9C9C"}
    options = {"ents": ['FUNC', 'FORMULA', 'FORMAT', 'VAR', 'QUOT'], "colors": colors}

    # visualize in html page
    displacy.serve(docs, style="ent", port=5001, options=options)

    divs = []
    for doc in docs:
        divs.append(spacy.displacy.render(doc, style="ent", options=options))
    return divs

# get spec_list
test = pd.read_excel('data/adam_adsl_spec.xlsx', sheet_name='ADSL')
spec_list = test['Derivation / Comment'].tolist()
spec_list = [sample_preprocess(s) for s in spec_list]

divs = ner_vis_divs(spec_list=spec_list, model_path='AZ/model-best-sm/')
test['div'] = divs
test.to_excel('AZ/data/adam_adsl_spec_divs.xlsx')


