import re

import pandas as pd


def sentence_split(spec: str):
    sentences = re.split('(\n|;|\. )', spec)
    seps = ['', '. ', '\n', ';', ' ']

    cats = []
    cat = 0
    next_flag = 0
    for index, s in enumerate(sentences):
        if s in seps:
            cats.append(cat)
            continue
        s = s.strip()
        if next_flag == 1 or s[:4].lower() in ['then', 'elif', 'else']:
            cats.append(cat)
            next_flag = 0
            if s[-1] == ':':
                next_flag = 1
            continue
        if s[-1] == ':':
            next_flag = 1
            cat += 1
            cats.append(cat)
            continue
        cat += 1
        cats.append(cat)

    for i in range(1, len(sentences)):
        if cats[i] != cats[i - 1]:
            if sentences[i - 1] == '\n' and i - 2 > 0:
                sentences[i - 2] += '@@@@'
            else:
                sentences[i - 1] += '@@@@'
    return ''.join(sentences)


def insertSeparator(spec:str):
    match = re.finditer('for records from|for [^;\n]{0,20} records', spec, re.IGNORECASE)
    inds = []
    for i in match:
        span = list(i.span())[0]
        inds.append(span)
    n = len(inds)
    if n <= 1:
        return spec
    res = ''
    for i in range(len(inds) - 1):
        res += spec[inds[i]:inds[i + 1]] + '@@@@'
    res += spec[inds[-1]:]
    return res


s1='''For patient didn't receive treatment, but had randomization code:
set to "ASSIGNED, NOT TREATED";
For patient completed study with status of "Screen failure":
set to "SCREEN FAILURE"
For patient who didn't have randomization code and study completion status is not "Screen failure":
set to "NOT ASSIGNED";'''

s2 = '''Calculation of DMDY = (Numeric version of DMDTC - Numeric version of RFSTDTC). 
Note: Add 1 if calculation is greater than or equal to 0.  If partial date then set to blank.  
'''

s3 = '''e.g. 'Read in dummy randomization data AstraZeneca_D8850C00008_Interim_dumrand_DUMMY_DDMMMYYYY as RANDTRT, then map with RAW.DM by RANDTRT.SUBJECT = RAW.DM.SUBJECT. 
if RANDTRT.TRTCODE = 'A' then ARMCD="A01";
else if RANDTRT.TRTCODE = 'B', then ARMCD="P01";'''

s4 = '''
'For RAW.PREG records: set to ='HCG' ;

For RAW.LB records:
Get the value of LBTESTCD by merging AZTESTCD in LBREF_MSTR_2021XXXX.xlsx where STUNITFL="Y" with the corresponding derived AZTESTCD value below in the corresponding datasets - 'L' concatenated with LBTEST in RAW.LB, if RAW.LB.LBPERF="C49488"; else SDTM.LB.LBTESTCD="LBALL".
 
For RAW.CLAB records:
Get the value of LBTEST by merging AZTESTCD in LBREF_MSTR_XXXXXXXX.xlsx where STUNITFL="Y" with the corresponding derived AZTESTCD value below in the corresponding datasets - 'L' concatenated with LBCTSTCD in RAW.CLAB.
'''

sentence_split(s1)
sentence_split(s2)
sentence_split(s3)
insertSeparator(s4)
import pandas as pd
data = pd.read_excel('AZ/data/D8850C00008_AEVSDMLB_20220629.xlsx', sheet_name='LB')
data['Unnamed: 7'] = data['Unnamed: 7'].fillna('')
data['pre'] = data['Unnamed: 7'].apply(insertSeparator)
data.to_excel('AZ/data/a.xlsx', index=False)

