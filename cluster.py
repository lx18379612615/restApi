import traceback

import bottle
import json
import demjson
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import config_default
from Logger import Logger

configs = config_default.configs
from connect_unit import connect_es_scroll
import DataResponse
from DataPrehandle import clean_x_datas

log = Logger(logname="app.log", logger="cluster").getlog()

@bottle.post()
def predict():
    log.info("predict start")
    log.info("request params: {}".format(bottle.request.params))
    params = bottle.request.params
    log.info("params: {}".format(params.__dict__.values()))
    # columns: 聚类分析的标签信息（标签中文名，标签英文名，标签状态）
    columns = bottle.request.forms.getunicode('columns')
    print(columns)
    # body: 客群的es条件
    body = bottle.request.forms.getunicode('sql_str')

    # newcolumns: 标签英文名-标签中文名
    newcolumns = getChiColumn(columns)
    log.info("convert english columns{} => chinese columns{}".format(columns, newcolumns))

    # full_data: 客群全部用户的聚类标签值
    full_data = connect_es_scroll(body)
    log.info("get es data {} => {} records".format(body, full_data.size))
    log.info(full_data)

    log.info("convert english columns {}".format(full_data.columns))
    # collist: 标签中文名
    collist = []
    for colname in full_data.columns:
        collist.append(newcolumns.get(colname))
    #full_data.columns = collist
    log.info("to chinese columns {}".format(collist))

    if full_data.empty or len(full_data.columns) == 0:
        log.info("dataset is empty, return back")
        back = DataResponse.DataResponse('', 1001, '当前条件聚类没有意义，请换成其他客群或标签')
        return json.dumps(back.__dict__, ensure_ascii=False)

    log.info("clean data set ...")
    data = clean_x_datas(full_data)

    try:
        log.info("train data set ...")
        clf, k = train(data)
    except ValueError as err:
        log.error("ERROR OCCURRED WHEN train data set : {}".format(err))
        log.error(traceback.format_exc())
        back = DataResponse.DataResponse('', 1001, '当前条件聚类没有意义，请换成其他客群或标签')
        return json.dumps(back.__dict__, ensure_ascii=False)

    log.info("train ok, predict itself ...")
    pred = clf.predict(data)
    log.info("pred = {}".format(pred))

    full_data.columns = collist
    r = pd.concat([full_data, pd.Series(pred, index=full_data.index)], axis=1)
    r.columns = list(full_data.columns) + [u'聚类类别']
    pList = []
    for i in range(k):
        cluster = crowdingLevel(full_data[r[u'聚类类别'] == i], columns)
        my_dict = cluster.to_dict('records')
        pList.append(my_dict)
    back = DataResponse.DataResponse(pList, '0', 'success')
    result = json.dumps(back.__dict__, ensure_ascii=False)
    log.info("result=" + result)
    return result

def getChiColumn(jsonStr):
    list = demjson.decode(jsonStr)
    columns = {}
    for d in list:
        columns[d["c_tag_english_name"]] = d["c_tag_category_name"]
    return columns

def train(data):
    clf, k = bestModel(data)
    return clf, k


# 训练模型，获得最佳的KMeans函数和聚类中心的个数
def bestModel(data):
    best_clf = None
    best_k = None
    best_score = None
    log.info("data: {}".format(data))
    for k in range(2, 5):
        log.info("k: {}".format(k))
        clf = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan')
        clf.fit(data)
        # silhouette_score(): 轮廓系数函数，有什么用？score是啥？
        score = silhouette_score(data, clf.labels_)
        if (best_score == None):
            best_clf = clf
            best_k = k
            best_score = score
        elif (best_score < score):
            best_clf = clf
            best_k = k
            best_score = score
    return best_clf, best_k


def crowdingLevel(data, jsonStr):
    list = demjson.decode(jsonStr)
    log.info(data)
    log.info(data.dtypes)
    data = data.astype('float64')
    i = 0
    r1 = pd.DataFrame(columns=('name', 'level', 'count', 'en_name', 'key_number', 'status'))
    enList = {}
    for ix, col in data.iteritems():
        for one in list:
            if one['c_tag_category_name'] == ix:
                item = one
        log.debug('col {}   {}'.format(type(col.values[0]),type(col.values[0])==np.int64))
        log.debug('col index {}   values  {} '.format(col.index,col.values))
        if (item['tag_status'] == 0 and len(item['dicts']) > 0):
            dic = {}
            for types in item['dicts']:
                c_sum=0
                if(type(col.values[0])==np.int64):
                    c_sum=np.sum(col == int(types['c_sub_entry']))
                elif(type(col.values[0])==np.float64):
                    c_sum=np.sum(col == float(types['c_sub_entry']))
                else:
                    c_sum=np.sum(col == types['c_sub_entry'])
                dic[types['c_dict_prompt']] = c_sum / len(col)
            dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
            name = item['c_tag_category_name'] + ' : '
            key_number = ''
            for keyItem in dic[:3]:
                log.info('keyItem  {}   {}'.format(keyItem[0],keyItem[1]))
                if keyItem[1] > 0:
                    name += str(keyItem[0]) + ' : ' + format_percent(keyItem[1]) + ','
                    log.info('name {}'.format(name))
                    for types in item['dicts']:
                        if types['c_dict_prompt'] == keyItem[0]:
                            key_number += types['c_sub_entry'] + ','
            name = name[:-1]
            key_number = key_number[:-1]
            level = dic[0][1]
            count = len(col)
            en_name = item['c_tag_english_name']
            status = item['tag_status']
            r1.loc[item['c_tag_category_name']] = [name, level, count, en_name, key_number, status]
            data = data.drop(ix, axis=1)
        else:
            enList.update({item['c_tag_category_name']: item['c_tag_english_name']})
        i += 1
    ############
    r = pd.DataFrame(columns=['name', 'level', 'count', 'en_name', 'key_number', 'status'])
    #log.info('++++++++++++++++++ r {}'.format(r))
    log.info(data)
    if len(data.columns) != 0:
        # r2 = pd.DataFrame(columns=['name', 'level', 'count', 'en_name', 'key_number'])
        list = []
        for col in data.columns:
            p1 = data[str(col)].quantile(0.025)
            p1 = round(p1, 4)
            p2 = data[str(col)].quantile(0.975)
            p2 = round(p2, 4)
            pen_name = enList.get(col)
            pmax = data[str(col)].max()
            pmax = round(pmax, 4)
            pmin = data[str(col)].min()
            pmin = round(pmin, 4)
            pkey_number = str(p1) + ',' + str(p2)
            if p2 == p1 or pmax == pmin:
                level = 0
            else:
                level = (p2 - p1) / (pmax - pmin)
            if p1 == p2:
                pname = col + '=' + str(p2)
            else:
                pname = str(p1) + '<=' + col + '<=' + str(p2)
            pcount = data.count()
            x = [pname, level, pcount[str(col)], pen_name, pkey_number]
            list.append(x)
        r2 = pd.DataFrame(list, columns=['name', 'level', 'count', 'en_name', 'key_number'])
        r2['status'] = 1
        r = r2
    r = pd.concat([r, r1])
    r = r.sort_values(by=['level'])
    num = 0
    for index in r.index:
        num += 1
        r.loc[index, 'value'] = num
    return r


def format_percent(num):
    return str(round(num * 100, 2)) + '%'




