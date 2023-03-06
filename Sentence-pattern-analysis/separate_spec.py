import re


def sentence_split(spec: str):
    sentences = re.split('(\n|;|\. )', spec)
    seps = ['', '. ', '\n', ';', ' ']

    cats = []
    cat = 0
    next_flag = 0
    for index, s in enumerate(sentences):
        try:
            s_next = sentences[index + 2]
        except:
            s_next = 'Unknow'
        if s in seps:
            cats.append(cat)
            continue
        s = s.strip()
        if next_flag == 1 or s[:4].lower() in ['then', 'elif', 'else']:
            cats.append(cat)
            next_flag = 0
            if s[-1] in [':', ','] or (s_next.strip()[:1].islower() and s not in ['. ', ';']):
                next_flag = 1
            continue
        if s[-1] in [':', ',', '']:
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
For records from RAW.CLAB then set to 'L'  concatenated with unformatted value of RAW.CLAB.LBCTSTCD;
For records from RAW.LB, set to 'L' concatenated with unformatted value of non-missing RAW.LB.LBTEST.
For RAW.PREG records: set to ='HCG' ;
For RAW.LB records:
Get the value of LBTESTCD by merging AZTESTCD in LBREF_MSTR_2021XXXX.xlsx where STUNITFL="Y" with the corresponding derived AZTESTCD value below in the corresponding datasets - 'L' concatenated with LBTEST in RAW.LB, if RAW.LB.LBPERF="C49488"; else SDTM.LB.LBTESTCD="LBALL".
For RAW.CLAB records:
Get the value of LBTEST by merging AZTESTCD in LBREF_MSTR_XXXXXXXX.xlsx where STUNITFL="Y" with the corresponding derived AZTESTCD value below in the corresponding datasets - 'L' concatenated with LBCTSTCD in RAW.CLAB.
'''
s5 = '''
Derived from DM.RACE: 
1=WHITE, 
2=BLACK OR AFRICAN AMERICAN, 
3=ASIAN, 
4=NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER, 
5=AMERICAN INDIAN OR ALASKA NATIVE, 
6=OTHER'''
s6='''For records of FORMULA59 and FORMULA60:\nif FORMULA61 then FORMULA62\nif FORMULA63 then FORMULA64\nif VAR73 is null then FORMULA65'''

sentence_split(s1)
sentence_split(s2)
sentence_split(s3)
sentence_split(s5)
sentence_split(s6)
insertSeparator(s6)





