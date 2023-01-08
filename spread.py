# -*- coding: utf-8 -*-
import json
import bottle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
import config_default
from collections import Counter
configs = config_default.configs
import connect_unit
import traceback
import DataResponse
from DataPrehandle import clean_x_datas
from DataPrehandle import clean_y_datas
from Logger import Logger

log = Logger(logname="app.log", logger="spread")

# 每个文件最大50M，最多5个
#handler = log.handlers.RotatingFileHandler(configs.get('log_file'), maxBytes=50*1024*1024, backupCount=5)
#handler.setLevel(log.DEBUG)
#handler.setFormatter(log.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'))
#log.getLogger('').addHandler(handler)

key_field_name = configs.get('key_field_name')

# 词云展示，其实也就是挑选出类似的可以进行扩散的标签
def get_main_feature():
    body = bottle.request.forms.getunicode('body')
    tag = bottle.request.forms.getunicode('tag')
    group_id = bottle.request.forms.getunicode('group_id')
    num = bottle.request.forms.getunicode('num')
    status = bottle.request.forms.getunicode('status')
    log.info("request body param: " + body)
    log.info("request tag param: " + tag)
    log.info("request group_id param: " + group_id)
    log.info("request num param: " + num)
    log.info("request status param: " + status)

    log.info("get data from es ...")
    try:
        df = connect_unit.connect_es(body)
        if df.empty:
            log.info("data is empty, return error code 1000")
            error_response = json.dumps(DataResponse.DataResponse('', 1000, '客群有效数据集为空').__dict__, ensure_ascii=False)
            return error_response
    except Exception as e:
        log.error("get data from es ERROR:", e)
        log.error(traceback.format_exc())
        error_response = json.dumps(DataResponse.DataResponse('', 1001, '获取客群数据超时，请减少客群数据大小').__dict__, ensure_ascii=False)
        return error_response
    log.info("get data from es complete")

    df2 = clean_x_datas(df)

    log.info(df2)
    if df.empty:
        log.info("after drop error columns, data is empty, return error code 1000")
        error_response = json.dumps(DataResponse.DataResponse('', 1000, '客群有效数据集为空').__dict__, ensure_ascii=False)
        return error_response

    log.info("df is not empty size={}, continue call local_get_main_feature(df) ...".format(df.size))
    result = local_get_main_feature(df2, df.columns)
    log.info("result of local_get_main_feature(df): " + json.dumps(result))

    response = json.dumps(DataResponse.DataResponse(result, 0, 'success').__dict__, ensure_ascii=False)
    log.info("response = {}".format(response))
    return response


def local_get_main_feature(x_train, named_cols):
    data_length = len(x_train)
    start_data = int(data_length * 0.025)
    end_data = int(data_length * 0.975)
    col = x_train.columns
    log.info("x_train: before format_feature")
    log.info(x_train)
    #pca = PCA(n_components='mle', svd_solver='full')
    pca = PCA(n_components=0.8, svd_solver='auto')
    pca.fit(x_train)
    component = pca.components_
    log.info("PCA component: after pca.fit(x_train)")
    log.info(component)
    my_percent = pca.explained_variance_ratio_
    log.info("my_percent: pca.explained_variance_ratio_")
    log.info(my_percent)
    #########################################
    yy = 0.0
    for xx in my_percent:
        yy = yy + xx
    log.info("yy:%s" % yy)
    #########################################
    position = np.apply_along_axis(get_max_index, 1, component)
    log.info("position: np.apply_along_axis(np.argmax(np.abs(data)), 1, component)")
    log.info(position)
    x_return = ["0"] * len(position)
    x_in_array = ["0"] * len(position)
    i = 0
    j = 0
    while i < len(position):
        if j > 0 and x_in_array.__contains__(position[i]):
            i = i + 1
            continue
        x_sort = pd.DataFrame({"sort": x_train[col[position[i]]]})
        x_sort = x_sort.sort_values(by=["sort"], ascending=True)
        start_value = x_sort.iloc[start_data: start_data + 1].values[0][0]
        end_value = x_sort.iloc[end_data: end_data + 1].values[0][0]
        count = Counter(x_train[col[position[i]]])
        m = max(Counter(x_train[col[position[i]]]).values())
        max_num = sorted([x for(x, y) in count.items() if y == m])[0]
        percent = float(m) / float(len(x_train)) * 100
        percent = float('%.2f' % percent)
        x_return[j] = named_cols[position[i]] + ":" + str(max_num) + ":" + str(percent) + ":" + str(start_value) + ":" + str(end_value)
        x_in_array[j] = position[i]
        i = i + 1
        j = j + 1
    return_array = ["0"] * 10
    return_array[0] = sorted(x_return[0: j], key=lambda x: float(x.split(":")[2]))
    return_array[1] = yy
    log.info("return_array:")
    log.info(return_array)
    return return_array


def get_max_index(data):
    return np.argmax(np.abs(data))


def population_spread():
    log.info("request params: {}".format(bottle.request))

    body = bottle.request.forms.getunicode('body')
    tag = bottle.request.forms.getunicode('tag')
    wait_body = bottle.request.forms.getunicode('wait_body')
    num = bottle.request.forms.getunicode('num')
    group_id = bottle.request.forms.getunicode('group_id')
    wait_group_id = bottle.request.forms.getunicode('wait_group_id')
    status = bottle.request.forms.getunicode('status')
    percent = bottle.request.forms.getunicode('percent')

    x_train = connect_unit.connect_es(body)
    log.info("x_train: body => es => x_train")
    log.info(x_train)
    if x_train.empty:
        log.warn("train data set is empty!!!")
        error_response = json.dumps(DataResponse.DataResponse('', 1000, '客群有效数据集为空').__dict__, ensure_ascii=False)
        return error_response

# 获取PAC之后的标签df
# body中直接筛选出10个wait_tag和一个tag
    log.info("wait_body:")
    log.info(wait_body)
    log.info("tag:")
    log.info(tag)
    if tag not in x_train.columns:
        log.error("tag column {} is removed".format(tag))
        error_response = json.dumps(DataResponse.DataResponse('', 3000, '标签列{}数据异常,应该为数字型'.format(tag)).__dict__, ensure_ascii=False)
        return error_response
    y_train = x_train[tag]
    log.info("y_train")
    log.info(y_train)
    y_train, status = clean_y_datas(y_train, status)
    log.info("y_train")
    log.info(y_train)
    x_train_wait = x_train.drop([tag], axis=1)
    x_train_wait = clean_x_datas(x_train_wait)
    # 处理一下y_train的行数
    y_train = y_train[:len(x_train_wait)]
    log.info("x_train_wait")
    log.info(x_train_wait)
    clf = bys(x_train_wait, y_train)
    x_test = connect_unit.connect_es_scroll(wait_body)
    if x_test.empty:
        error_response = json.dumps(DataResponse.DataResponse('', 1000, '测试集合有效数据集为空').__dict__, ensure_ascii=False)
        return error_response

    x_test_wait = x_test.drop([key_field_name, "client_name", "mobile_tel"], axis=1)
    x_test_wait = clean_x_datas(x_test_wait)
#这里不在使用predict方法去计算扩散是否准确，只使用predict_proba
    y_predict = clf.predict(x_test_wait)
#这里用于得到这里一共有多少字典项，并排序，给后面的x_percent_df使用
    #x_percent_df_columns = np.unique(y_train).astype('str')
    x_percent = clf.predict_proba(x_test_wait)
    x_percent_df = pd.DataFrame(x_percent)
    #x_percent_df.columns = x_percent_df_columns
    y_percent = [0] * len(x_percent)
    i = 0
    j = 0
    log.info("status:")
    log.info(status)
    log.info("percent:%s" % percent)
    log.info("x_percent_df.columns")
    log.info(x_percent_df.columns)
    log.info("x_percent_df.loc")
    log.info(x_percent_df.loc)
    if x_percent_df.columns.__contains__(status):
        while i < y_predict.size:
            if x_percent_df.loc[i, status] >= float(percent):
            # y_predict[i] = 1
                info = {key_field_name: x_test[key_field_name][i], "client_name": x_test["client_name"][i], "mobile_tel": x_test["mobile_tel"][i], "percent": float("%.2f" % (x_percent_df.loc[i, status]*100))}
                y_percent[j] = info
                j = j + 1
        # else:
        #     y_predict[i] = 0
            i = i + 1
    y_percent = y_percent[:j]
    y_percent.sort(key=tag_percent_value, reverse=True)
    log.info("x_percent:")
    log.info(x_percent)
    log.info("y_percent:")
    log.info(y_percent)

 #   try:
 #       redis_conn = connect_unit.redis_cluster()
 #       redis_key_percent = "population_spread" + "." + group_id + "." + wait_group_id + "." + num + "." + tag + "." + status + "." + percent + ".percent"
 #       redis_conn.set(redis_key_percent, y_percent, ex=1800)
 #   except:
 #       log.error("save data into redis ERROR:")
 #       log.error(traceback.format_exc())

    data_response = json.dumps(DataResponse.DataResponse(y_percent, 0, 'success').__dict__, ensure_ascii=False)
    log.info("dataResponse = {}".format(data_response))
    return data_response



def tag_percent_value(elem):
    return elem.get('percent')


def bys(x_train, y_train):
    log.info("train by GaussianNB model:")
    log.info("x_train:")
    log.info(x_train)
    log.info("y_train:")
    log.info(y_train)
    clf = GaussianNB().fit(x_train, y_train)
    log.info("clf:")
    log.info(clf)
    return clf


def format_bit_array(array):
    bit_array = []
    n = 0
    for i in range(len(array)):
        n <<= 1
        n += int(array[i])
        if i & 0x07 == 7:
            bit_array.append(n)
            n = 0
    if len(array) & 0x07 != 0:
        n <<= 8 - (len(array) % 8)
        bit_array.append(n)
    return bit_array

