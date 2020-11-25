# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import os
import string
import time

current_path = os.path.dirname(__file__)

# %%
def get_dic(current_path):
    dic = []
    data_dir = current_path + "/data"
    for s in os.listdir(data_dir):
        newDir=os.path.join(data_dir,s)
        file = open(newDir, 'r', encoding='utf-8')
        for line in file.readlines():
            word = line.strip().split()
            dic.extend(word)
        file.close()
    dic = set(dic)
    diff = {'%', '.'}
    for i in range(10000):
        diff.add(str(i))
    for letter in string.ascii_letters:
        diff.add(letter)
    dic = dic - diff
    return dic

# %%
def get_win_size(dic):
    return max(map(len, [w for w in dic]))

# %%
def FMM(dic, text, window_size):
    result = []
    label = []
    index = 0
    found = 0
    text_size = len(text)
    while text_size > index:
        for size in range(window_size, 0,  -1):
            piece = text[index: size+index]
            if piece in dic:
                found = 1
                index += size - 1
                break
        index = index + 1
        result.append(piece)
        label.append(found)
        found = 0
    return (result, label)

# %%
def RMM(dic, text, window_size):
    result = []
    label = []
    index = len(text)
    found = 0
    window = min(window_size, index)
    while index > 0:
        for size in range(window, 0,  -1):
            piece = text[index-size: index]
            if piece in dic:
                found = 1
                index -= size - 1
                break
        index = index - 1
        result.append(piece)
        label.append(found)
        found = 0
    result.reverse()
    label.reverse()
    return (result, label)

# %%
def BIMM(dic, text, window_size):
    (res_fmm, fmm_label) = FMM(dic, text, window_size)
    (res_rmm, rmm_label) = RMM(dic, text, window_size)
    if len(res_fmm) == len(res_rmm):
        if res_fmm == res_rmm:
            return (res_rmm, rmm_label)
        else:
            f_count = len([w for w in res_fmm if len(w) == 1])
            r_count = len([w for w in res_rmm if len(w) == 1])
            return (res_fmm, fmm_label) if f_count < r_count else (res_rmm, rmm_label)
    else:
        return (res_fmm, fmm_label) if len(res_fmm) < len(res_rmm) else (res_rmm, rmm_label)

def tidy(result, label):
    ##### 合并
    # 姓氏
    fmy_name_file = open(current_path + "/zh_family_names.txt", 'r', encoding='utf-8')
    fmy_name = fmy_name_file.readline()
    fmy_name = fmy_name.strip().split()
    # 量词
    quantifier = ['千','万','亿','多万','时','分','余万','日','月','月份','月底','月末', '世纪']
    l = len(result)
    ret = []
    tmp = ''
    # for i in range(l):
    #     label[i] = 1
    for i in range(l):
        # 英文和数字
        if (result[i].encode('UTF-8').isalpha() and result[i+1].encode('UTF-8').isalpha()) or (result[i][-1].encode('UTF-8').isdigit() and result[i+1][0].encode('UTF-8').isdigit()):
            # print(result[i], result[i+1])
            label[i] = 0
            label[i+1] = 0
        # 几个标点符号前后
        elif (i < l-1) and (result[i].encode('UTF-8').isdigit()) and (result[i+1][0] in ['∶',':','.','：']):
            label[i] = 0
            label[i+1] = 0
        elif (i < l-1) and (result[i][-1] in ['∶',':','.','：']) and (result[i+1][0].encode('UTF-8').isdigit()):
            label[i] = 0
            label[i+1] = 0
        elif (i < l-1) and (result[i].encode('UTF-8').isalpha()) and (result[i+1] == '.'):
            label[i] = 0
            label[i+1] = 0
        # 百分号前后
        elif (i < l-1) and (result[i][-1].encode('UTF-8').isdigit()) and (result[i+1] == '%'):
            label[i] = 0
            label[i+1] = 0
        # 姓氏
        elif (i < l-1) and (result[i] in fmy_name):
            if (i < l-2) and len(result[i+1]) == 1 and len(result[i+2]) == 1:
                label[i] = 0
                label[i+1] = 0
                label[i+2] = 0
        # 量词
        elif (i < l-1) and (result[i][-1].encode('UTF-8').isdigit()) and (result[i+1] in quantifier):  ## 顺序问题
            label[i] = 0
            label[i+1] = 0
        elif (i < l-1) and (result[i].encode('UTF-8').isdigit() and int(result[i]) > 1000) and (result[i+1] == '年'):
            label[i] = 0
            label[i+1] = 0
        # '&'前后
        elif (i < l-1) and (result[i] == '&' or result[i+1] == '&'):
            label[i] = 0
            label[i+1] = 0
        # '-'前后
        elif (i > 0 and i < l-1) and (result[i] == '-' or result[i+1] == '-'):
            label[i] = 0
            label[i+1] = 0
        elif (i < l-1 and result[i].encode('UTF-8').isalpha() and result[i+1] == '-3') or (i < l-1 and result[i] == '-3' and result[i+1].encode('UTF-8').isalpha()):
            label[i] = 0
            label[i+1] = 0
        # ','前后数字
        elif i < l-2 and result[i].encode('UTF-8').isdigit() and result[i+1] == ',' and result[i+2].encode('UTF-8').isdigit():
            label[i] = 0
            label[i+1] = 0
            label[i+2] = 0
        # @前后英文
        elif i < l-2 and result[i].encode('UTF-8').isalpha() and result[i+1] == '@' and result[i+2].encode('UTF-8').isalpha():
            label[i] = 0
            label[i+1] = 0
            label[i+2] = 0
        # “～”前后数字
        elif (i < l-2) and result[i][-1].encode('UTF-8').isdigit() and result[i+1] == '～' and result[i+2][0].encode('UTF-8').isdigit():
            label[i] = 0
            label[i+1] = 0
            label[i+2] = 0
        # 连接'·'和其前后的中文(姓名)
        elif i < l-2 and (result[i+1] == '·' or result[i+1] == '•'):
            label[i] = 0
            label[i+1] = 0
            label[i+2] = 0
        # 合并低频词
        if label[i] <= 31 and len(result[i]) == 1:
            if tmp == '':
                tmp = result[i]
            else:
                tmp = tmp + result[i]
            if i == l-1:
                ret.append(tmp)
        else:
            if tmp != '':
                ret.append(tmp)
                tmp = ''
            ret.append(result[i])
    
    ##### 分割:
    l = len(ret)
    for i in range(l):
        if ret[i] == '↓↓':
            ret.insert(i+1, ret[i][1:])
            ret[i] = '↓'

    if ''.join(ret) != ''.join(result):
        print(result)
        print(label)
        print(ret)
    return ret

# %%
def main():
    start_time = time.time()
    dic = get_dic(current_path)
    window_size = get_win_size(dic)

    test = open(current_path + "/test.txt", 'r', encoding='utf-8')
    submit = open(current_path + "/181220067.txt", 'w', encoding='utf-8')
    results = []
    result_list = []
    labels = []
    result_dic = {}
    for line in test.readlines():
        (result, label) = BIMM(dic, line, window_size)
        results.append(result)
        result_list.extend(result)
        labels.append(label)
    result_set = set(result_list)
    for itm in result_set:
        result_dic[itm] = result_list.count(itm)
    for i in range(len(results)):
        result = results[i]
        label = labels[i]
        for j in range(len(result)):
            label[j] = result_dic[result[j]]
        result = tidy(result, label)
        result = ' '.join(result)
        submit.write(result)
    submit.close()
    test.close()
    finish_time = time.time()
    total_time = finish_time - start_time
    print("total_time:", total_time)


if __name__ == "__main__":
    main()