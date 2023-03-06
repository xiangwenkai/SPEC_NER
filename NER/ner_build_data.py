import time
import numpy as np
import pandas as pd
from random import choice
import re
import random
from spacy.tokens import DocBin
from tqdm import tqdm
import spacy

'''
1.不影响结果的空格，应该设置为随机有或者无
2.
'''


def get_entity_set(ll):
    '''
    from some string dicts to a union set of values in the dicts
    :param ll:
    :return:
    '''
    res = []
    for x in ll:
        x = eval(x)
        temp = list(x.values())
        res.extend(temp)
    res = list(set(res))
    return res


def get_one_formula(formulas, vars, quots, funcs):
    '''
    random generate a formula string from formulas
    :param formulas:
    :param vars:
    :param quots:
    :return:
    '''
    # f = '[V1] / [V11] = [quot1]'
    fo = choice(formulas)
    var_reps = re.findall('\[V[0-9]+\]', fo)
    quot_reps = re.findall('\[quot[0-9]+\]', fo)
    fc_reps = re.findall('\[fc[0-9]+\]', fo)

    all_reps = var_reps + quot_reps + fc_reps
    if len(all_reps) == 0:
        return fo, []

    reps_cat = ['VAR']*len(var_reps) + ['QUOT']*len(quot_reps) + ['FUNC']*len(fc_reps)
    inds_begin = [fo.find(x) for x in all_reps]
    inds_argsort = np.argsort(inds_begin)

    sub_labels = []
    for ind in inds_argsort:
        rep = all_reps[ind]
        cat = reps_cat[ind]
        ind_start = fo.find(rep)
        if cat == 'VAR':
            var = choice(vars)
            len_target = len(var)
            fo = fo.replace(rep, var)
            sub_labels.append((ind_start, ind_start + len_target, 'VAR'))
        if cat == 'QUOT':
            quot = choice(quots)
            len_target = len(quot)
            fo = fo.replace(rep, quot)
            sub_labels.append((ind_start, ind_start + len_target, 'QUOT'))
        if cat == 'FUNC':
            fc, _ = get_one_func(funcs, formulas, vars, quots)
            len_target = len(fc)
            fo = fo.replace(rep, fc)
            sub_labels.append((ind_start, ind_start + len_target, 'FUNC'))
    return fo, sub_labels


def get_one_func(funcs, formulas, vars, quots):
    '''
    random generate a function string from funcs
    :param funcs:
    :param formulas:
    :param vars:
    :param quots:
    :return:
    '''
    fu = choice(funcs)
    func_reps = re.findall('\[fc[0-9]+\]', fu)
    formula_reps = re.findall('\[formu[0-9]+\]', fu)
    var_reps = re.findall('\[V[0-9]+\]', fu)
    quot_reps = re.findall('\[quot[0-9]+\]', fu)

    all_reps = func_reps + formula_reps + var_reps + quot_reps
    if len(all_reps) == 0:
        return fu, []

    reps_cat = ['FUNC'] * len(func_reps) + ['FORMULA'] * len(formula_reps) + \
               ['VAR'] * len(var_reps) + ['QUOT'] * len(quot_reps)
    inds_begin = [fu.find(x) for x in all_reps]
    inds_argsort = np.argsort(inds_begin)

    sub_labels = []
    for ind in inds_argsort:
        rep = all_reps[ind]
        cat = reps_cat[ind]
        ind_start = fu.find(rep)
        if cat == 'FUNC':
            temp, sub_ = get_one_func(funcs, formulas, vars, quots)
            len_target = len(temp)
            fu = fu.replace(rep, temp)
            sub_labels.append((ind_start, ind_start + len_target, cat))
            if len(sub_) > 0:
                sub_ = [(i+ind_start, j+ind_start, s) for i, j, s in sub_]
                sub_labels.extend(sub_)
        if cat == 'FORMULA':
            temp, sub_ = get_one_formula(formulas, vars, quots, funcs)
            len_target = len(temp)
            fu = fu.replace(rep, temp)
            sub_labels.append((ind_start, ind_start + len_target, cat))
            if len(sub_) > 0:
                sub_ = [(i+ind_start, j+ind_start, s) for i, j, s in sub_]
                sub_labels.extend(sub_)
        if cat == 'VAR':
            var = choice(vars)
            len_target = len(var)
            fu = fu.replace(rep, var)
            sub_labels.append((ind_start, ind_start + len_target, cat))
        if cat == 'QUOT':
            quot = choice(quots)
            len_target = len(quot)
            fu = fu.replace(rep, quot)
            sub_labels.append((ind_start, ind_start + len_target, cat))
    '''
        for func_rep in func_reps:
        temp = get_one_func(funcs, formulas, vars, quots)
        fu = fu.replace(func_rep, temp)
    for formula_rep in formula_reps:
        temp = get_one_formula(formulas, vars, quots, funcs)
        fu = fu.replace(formula_rep, temp)
    for var_rep in var_reps:
        # if var_rep[0] == ' ':
        #     if random.random()>0.5:
        #         var_rep = var_rep[1:]
        # if var_rep[-1] == ' ':
        #     if random.random()>0.5:
        #         var_rep = var_rep[:-1]
        var = choice(vars)
        fu = fu.replace(var_rep, var)
    for quot_rep in quot_reps:
        quot = choice(quots)
        fu = fu.replace(quot_rep, quot)
    '''
    return fu, sub_labels


def get_one_sample_old(rep_samples, vars, quots, formats, formulas, funcs):
    '''
    random generate an not nested NER sample.
    :param rep_s:
    :return:
    '''
    cats = ['FUNC', 'FORMULA', 'FORMAT', 'VAR', 'QUOT']
    sample = choice(rep_samples)
    func_reps = re.findall('\[fc[0-9]+\]', sample)
    formula_reps = re.findall('\[formu[0-9]+\]', sample)
    format_reps = re.findall('\[format[0-9]+\]', sample)
    var_reps = re.findall('\[V[0-9]+\]', sample)
    quot_reps = re.findall('\[quot[0-9]+\]', sample)

    # 每类的数量
    cats_num = [len(func_reps), len(formula_reps), len(format_reps), len(var_reps), len(quot_reps)]

    if sum(cats_num) == 0:
        return get_one_sample_old(rep_samples, vars, quots, formats, formulas, funcs)

    all_reps = func_reps+formula_reps+format_reps+var_reps+quot_reps

    # 目标字符类别列表
    labels_cat = []
    for cat, number in zip(cats, cats_num):
        labels_cat.extend([cat]*number)

    ind_begin = [sample.find(x) for x in all_reps]
    arg_sort_begin = np.argsort(ind_begin).tolist()
    len_rep = [len(x) for x in all_reps]

    len_target = []

    for func_rep in func_reps:
        temp, _ = get_one_func(funcs, formulas, vars, quots)
        len_target.append(len(temp))
        sample = sample.replace(func_rep, temp)

    for formula_rep in formula_reps:
        temp, _ = get_one_formula(formulas, vars, quots, funcs)
        len_target.append(len(temp))
        sample = sample.replace(formula_rep, temp)

    for format_rep in format_reps:
        temp = choice(formats)
        len_target.append(len(temp))
        sample = sample.replace(format_rep, temp)

    for var_rep in var_reps:
        # if var_rep[0] == ' ':
        #     if random.random() > 0.5:
        #         var_rep = var_rep[1:]
        # if var_rep[-1] == ' ':
        #     if random.random() > 0.5:
        #         var_rep = var_rep[:-1]
        var = choice(vars)
        len_target.append(len(var))
        sample = sample.replace(var_rep, var)

    for quot_rep in quot_reps:
        quot = choice(quots)
        len_target.append(len(quot))
        sample = sample.replace(quot_rep, quot)

    if len(len_rep) == 1:
        return [sample, {'entities': [(ind_begin[0], ind_begin[0]+len_target[0], labels_cat[0])]}]

    # 计算排序后的字符长度差值
    argsort_diff = []
    for ind in arg_sort_begin:
        argsort_diff.append(len_target[ind] - len_rep[ind])

    # 计算排序后的累计字符长度差值
    cum_diff = np.cumsum(argsort_diff).tolist()

    # 计算目标字符起始坐标
    target_begin = []
    for i in range(len(arg_sort_begin)):
        ind = arg_sort_begin.index(i)
        if ind == 0:
            target_begin.append(ind_begin[i])
        else:
            target_begin.append(ind_begin[i]+cum_diff[ind-1])

    # 计算目标字符起始和结束坐标
    # target_loc_list = [(x, x+len_target[i]) for i, x in enumerate(target_begin)]

    '''
    # 实体用字典表示
    labels_dict = {}
    k = 0
    for i, number in enumerate(cats_num):
        if number > 0:
            labels_dict[cats[i]] = target_loc_list[k:k+number]
            k += number
    '''
    # 实体用数组表示
    labels_list = []
    i = 0
    for x, label in zip(target_begin, labels_cat):
        labels_list.append((x, x+len_target[i], label))
        i += 1

    return (sample, {'entities': labels_list})


def get_one_sample_nested(rep_samples, vars, quots, formats, formulas, funcs, target_cat=['FUNC', 'FORMULA', 'FORMAT', 'VAR', 'QUOT']):
    '''
    random generate an nested NER sample.
    :param rep_s:
    :return:
    '''
    # cats = ['FUNC', 'FORMULA', 'FORMAT', 'VAR', 'QUOT']
    sample = choice(rep_samples)

    func_reps = re.findall('\[fc[0-9]+\]', sample)
    formula_reps = re.findall('\[formu[0-9]+\]', sample)
    format_reps = re.findall('\[format[0-9]+\]', sample)
    var_reps = re.findall('\[V[0-9]+\]', sample)
    quot_reps = re.findall('\[quot[0-9]+\]', sample)

    # 每类的数量
    cats_num = [len(func_reps), len(formula_reps), len(format_reps), len(var_reps), len(quot_reps)]

    if sum(cats_num) == 0:
        return get_one_sample_nested(rep_samples, vars, quots, formats, formulas, funcs)

    all_reps = func_reps+formula_reps+format_reps+var_reps+quot_reps


    reps_cat = ['FUNC'] * len(func_reps) + ['FORMULA'] * len(formula_reps) + \
               ['FORMAT'] * len(format_reps) + ['VAR'] * len(var_reps) + \
               ['QUOT'] * len(quot_reps)

    inds_begin = [sample.find(x) for x in all_reps]
    inds_argsort = np.argsort(inds_begin)

    sub_labels = []
    for ind in inds_argsort:
        rep = all_reps[ind]
        cat = reps_cat[ind]
        ind_start = sample.find(rep)
        if cat == 'FUNC':
            temp, sub_ = get_one_func(funcs, formulas, vars, quots)
            len_target = len(temp)
            sample = sample.replace(rep, temp)
            sub_labels.append((ind_start, ind_start + len_target, cat))
            if len(sub_) > 0:
                sub_ = [(i + ind_start, j + ind_start, s) for i, j, s in sub_]
                sub_labels.extend(sub_)
        if cat == 'FORMULA':
            temp, sub_ = get_one_formula(formulas, vars, quots, funcs)
            len_target = len(temp)
            sample = sample.replace(rep, temp)
            sub_labels.append((ind_start, ind_start + len_target, cat))
            if len(sub_) > 0:
                sub_ = [(i + ind_start, j + ind_start, s) for i, j, s in sub_]
                sub_labels.extend(sub_)
        if cat == 'FORMAT':
            temp = choice(formats)
            len_target = len(temp)
            sample = sample.replace(rep, temp)
            sub_labels.append((ind_start, ind_start + len_target, cat))
        if cat == 'VAR':
            var = choice(vars)
            len_target = len(var)
            sample = sample.replace(rep, var)
            sub_labels.append((ind_start, ind_start + len_target, cat))
        if cat == 'QUOT':
            quot = choice(quots)
            len_target = len(quot)
            sample = sample.replace(rep, quot)
            sub_labels.append((ind_start, ind_start + len_target, cat))
    # 筛选目标实体
    sub_labels = [(i, j, m) for i, j, m in sub_labels if m in target_cat]
    return (sample, {'entities': sub_labels})


def get_one_sample_unnested(rep_samples, vars, quots, formats, formulas, funcs, target_cat=['FUNC', 'FORMULA', 'FORMAT', 'VAR', 'QUOT']):
    '''
    random generate an nested NER sample.
    :param rep_s:
    :return:
    '''
    # cats = ['FUNC', 'FORMULA', 'FORMAT', 'VAR', 'QUOT']
    sample = choice(rep_samples)

    func_reps = re.findall('\[fc[0-9]+\]', sample)
    formula_reps = re.findall('\[formu[0-9]+\]', sample)
    format_reps = re.findall('\[format[0-9]+\]', sample)
    var_reps = re.findall('\[V[0-9]+\]', sample)
    quot_reps = re.findall('\[quot[0-9]+\]', sample)

    # 每类的数量
    cats_num = [len(func_reps), len(formula_reps), len(format_reps), len(var_reps), len(quot_reps)]

    if sum(cats_num) == 0:
        return get_one_sample_unnested(rep_samples, vars, quots, formats, formulas, funcs)

    all_reps = func_reps+formula_reps+format_reps+var_reps+quot_reps


    reps_cat = ['FUNC'] * len(func_reps) + ['FORMULA'] * len(formula_reps) + \
               ['FORMAT'] * len(format_reps) + ['VAR'] * len(var_reps) + \
               ['QUOT'] * len(quot_reps)

    inds_begin = [sample.find(x) for x in all_reps]
    inds_argsort = np.argsort(inds_begin)

    sub_labels = []
    for ind in inds_argsort:
        rep = all_reps[ind]
        cat = reps_cat[ind]
        ind_start = sample.find(rep)
        if cat == 'FUNC':
            temp, _ = get_one_func(funcs, formulas, vars, quots)
            len_target = len(temp)
            sample = sample.replace(rep, temp)
            sub_labels.append((ind_start, ind_start + len_target, cat))
            # if len(sub_) > 0:
            #     sub_ = [(i + ind_start, j + ind_start, s) for i, j, s in sub_]
            #     sub_labels.extend(sub_)
        if cat == 'FORMULA':
            temp, _ = get_one_formula(formulas, vars, quots, funcs)
            len_target = len(temp)
            sample = sample.replace(rep, temp)
            sub_labels.append((ind_start, ind_start + len_target, cat))
            # if len(sub_) > 0:
            #     sub_ = [(i + ind_start, j + ind_start, s) for i, j, s in sub_]
            #     sub_labels.extend(sub_)
        if cat == 'FORMAT':
            temp = choice(formats)
            len_target = len(temp)
            sample = sample.replace(rep, temp)
            sub_labels.append((ind_start, ind_start + len_target, cat))
        if cat == 'VAR':
            var = choice(vars)
            len_target = len(var)
            sample = sample.replace(rep, var)
            sub_labels.append((ind_start, ind_start + len_target, cat))
        if cat == 'QUOT':
            quot = choice(quots)
            len_target = len(quot)
            sample = sample.replace(rep, quot)
            sub_labels.append((ind_start, ind_start + len_target, cat))
    # 筛选目标实体
    sub_labels = [(i, j, m) for i, j, m in sub_labels if m in target_cat]
    return (sample, {'entities': sub_labels})



if __name__ == "__main__":
    data = pd.read_excel('AZ/data/res2.xlsx')
    print(f"columns: {data.columns}")

    # 获取样本公式集合
    vars = get_entity_set(data['Vars'].tolist())
    quots = get_entity_set(data['quotation'].tolist())
    formats = get_entity_set(data['format'].tolist())
    formulas = get_entity_set(data['formu_dict'].tolist())
    funcs = get_entity_set(data['funcs_dict'].tolist())

    df_vars = pd.DataFrame({'var': vars})
    df_vars.to_excel('/Users/wkx/Desktop/AZ/formula_extract/entity_raw/df_vars.xlsx', index=False)
    df_quots = pd.DataFrame({'quot': quots})
    df_quots.to_excel('/Users/wkx/Desktop/AZ/formula_extract/entity_raw/df_quots.xlsx', index=False)
    df_formats = pd.DataFrame({'format': formats})
    df_formats.to_excel('/Users/wkx/Desktop/AZ/formula_extract/entity_raw/df_formats.xlsx', index=False)
    df_formulas = pd.DataFrame({'formula': formulas})
    df_formulas.to_excel('/Users/wkx/Desktop/AZ/formula_extract/entity_raw/df_formulas.xlsx', index=False)
    df_funcs = pd.DataFrame({'func': funcs})
    df_funcs.to_excel('/Users/wkx/Desktop/AZ/formula_extract/entity_raw/df_funcs.xlsx', index=False)


    # get_one_formula(formulas, vars, quots, funcs)
    # get_one_func(funcs, formulas, vars, quots)

    # 随机挑选代号mask后的句子，随机法填充还原，并标记实体位置，形成样本
    rep_samples = list(set(data['res'].tolist()))

    # 测试样本生成效率
    t = time.time()
    n=100
    for _ in range(n):
        sample = get_one_sample_unnested(rep_samples)
    print(f"Average generate time {(time.time()-t)/n}")







