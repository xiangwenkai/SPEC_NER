from googletrans import Translator
import requests
requests.get(url='http://127.0.0.1:7204/onlineTranserver/serverapi/goTextTranslate',
             params={"field": "PV", "from": "CHINESE", "origin": "你好", "to": "ENGLISH"}
             )

requests.get(url='https://translation.googleapis.com/language/translate/v2',
             params={"field": "PV", "from": "CHINESE", "origin": "你好", "to": "ENGLISH"}
             )

translator = Translator(service_urls=['translate.googleapis.com'])
result = translator.translate('How are you?', dest='zh-CN')





import re
import html
from urllib import parse
import requests

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


print(translate("about your situation", "zh-CN", "en"))


