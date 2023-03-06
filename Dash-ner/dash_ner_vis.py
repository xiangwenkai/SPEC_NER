import dash
from dash.dependencies import Input, Output, State
import spacy
import requests
import numpy as np
import re
import html
from urllib import parse

GOOGLE_TRANSLATE_URL = 'http://translate.google.cn/m?q=%s&tl=%s&sl=%s'


def translate(text, to_language="zh-CN", text_language="en"):
    text = parse.quote(text)
    url = GOOGLE_TRANSLATE_URL % (text,to_language,text_language)
    response = requests.get(url, verify=False)
    data = response.text
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    result = re.findall(expr, data)
    if (len(result) == 0):
        return ""
    return html.unescape(result[0])


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
    re.sub('([A-Z]{1})\.[ ]{1}', lambda x: x.group(1) + ' . ', s)
    # 多个空格变成一个
    s = re.sub('[ ]+', ' ', s)
    return s


DEFAULT_LABEL_COLORS = {'FUNC': "#FF6A6A", "FORMULA": "#FF6A6A", "VAR": "#6495ED", "QUOT": "#9AFF9A", "FORMAT": "#9C9C9C"}


# Initialize the application
app = dash.Dash(__name__)
nlp = spacy.load("model-best-trf/")


def entname(name):
    return dash.html.Span(name, style={
        "font-size": "0.8em",
        "font-weight": "bold",
        "line-height": "1",
        "border-radius": "0.35em",
        "text-transform": "uppercase",
        "vertical-align": "middle",
        "margin-left": "0.5rem"
    })

def entbox(children, color):
    return dash.html.Mark(children, style={
        "background": color,
        "padding": "0.45em 0.6em",
        "margin": "0 0.4em",
        "line-height": "2.5",
        "border-radius": "0.35em",
    })


def entity(children, name):
    if type(children) is str:
        children = [children]

    children.append(entname(name))
    color = DEFAULT_LABEL_COLORS[name]
    return entbox(children, color)


def render(doc):
    children = []
    last_idx = 0
    for ent in doc.ents:
        children.append(doc.text[last_idx:ent.start_char])
        children.append(
            entity(doc.text[ent.start_char:ent.end_char], ent.label_))
        last_idx = ent.end_char
    children.append(doc.text[last_idx:])
    return children


app.layout = dash.html.Div(
    [
        dash.html.I("Please input a spec: "),
        dash.html.Br(),
        # dcc.Input(id="input_spec", type="text", placeholder="",
        #           style={'marginRight': '20px', 'width': "800px"}),
        dash.dcc.Textarea(
            id='input_spec',
            style={'width': '100%', 'height': 100},
        ),
        dash.html.Button(id='submit-button', type='submit', children='Submit'),
        dash.html.Button(id='translation-button', type='submit', children='translation'),
        dash.html.Br(),
        dash.html.Br(),
        dash.html.Div(id='output_div'),
        dash.html.Br(),
        dash.html.I("Direct translation: "),
        dash.dcc.Textarea(
            id='translation', className='translation',
            style={'width': '100%', 'height': 100},
        ),
        dash.html.Br(),
        # dash.html.Div(id='translation'),

        dash.html.I("Translation with NER process: "),
        dash.html.Br(),
        dash.dcc.Textarea(
            id='translation_ner',
            style={'width': '100%', 'height': 100},
        ),
        # dash.html.Div(id='translation_ner'),
    ]
)

@app.callback(Output('output_div', 'children'),
                  [Input('submit-button', 'n_clicks')],
                  [State('input_spec', 'value')],
                  )
def update_output(clicks, input_value):
    if clicks is not None:
        return render(nlp(input_value))

@app.callback(Output('translation', 'value'),
                  [Input('translation-button', 'n_clicks')],
                  [State('input_spec', 'value')],
                  )
def update_output(clicks, input_value):
    # 返回正常翻译结果
    if clicks is not None:
        print("start translate......")
        return translate(input_value, "zh-CN", "en")
        # req = requests.get(url='http://127.0.0.1:7204/onlineTranserver/serverapi/goTextTranslate',
        #                    params={"field": "PV", "from": "ENGLISH", "origin": input_value, "to": "CHINESE"}
        #                    )
        # return req['data']

@app.callback(Output('translation_ner', 'value'),
                  [Input('translation-button', 'n_clicks')],
                  [State('input_spec', 'value')],
                  )
def update_output(clicks, input_value):
    if clicks is not None:
        # NER替换-->翻译-->还原
        input_value = sample_preprocess(input_value)
        ents = [(ent.text, ent.start_char, ent.end_char, f'[AAAA{i}]') for i, ent in enumerate(nlp(input_value).ents)]

        rep = input_value
        pre_len = [len(i[0]) for i in ents]
        pre_argsort = np.argsort(pre_len)[::-1]
        for i in pre_argsort:
            rep = rep.replace(ents[i][0], ' '+ents[i][3]+' ')
        rep = rep.replace('  ', ' ')

        trans_rep = translate(rep, "zh-CN", "en")
        # trans = requests.get(url='http://127.0.0.1:7204/onlineTranserver/serverapi/goTextTranslate',
        #                    params={"field": "PV", "from": "ENGLISH", "origin": rep, "to": "CHINESE"}
        #                    )
        # trans_rep = trans['data']
        res = trans_rep
        for i in ents:
            res = res.replace(i[3], i[0])
        return res

def rep_to_raw(s, ents):
    for i in ents:
        s = s.replace(i[3], i[0])
    return s

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)


