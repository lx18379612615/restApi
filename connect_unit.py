# -*- coding: utf-8 -*-
import pymysql
import config_default
import numpy as np
import redis
from elasticsearch import Elasticsearch

from Logger import Logger

import pandas as pd
from pandasticsearch import Select
# from rediscluster import StrictRedisCluster
from redis.sentinel import Sentinel

configs = config_default.configs
log = Logger(logname="app.log", logger="connect_unit").getlog()
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 160)
# 不换行显示
pd.set_option('expand_frame_repr', False)
# 设置列左对齐
pd.set_option('display.colheader_justify', 'left')

# 每个文件最大50M，最多5个
# handler = logging.handlers.RotatingFileHandler(configs.get('log_file'), maxBytes=50*1024*1024, backupCount=5)
# handler.setLevel(logging.DEBUG)
# handler.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'))
# logging.getLogger('').addHandler(handler)


def redis_cluster():
    redis_nodes = configs.get('redis_nodes')
    sentinel_list = configs.get('sentinel_list')
    try:
        # if redis_nodes:
        #     redis_conn = StrictRedisCluster(startup_nodes=redis_nodes, password=configs.get('redis_password'))
        # elif sentinel_list:
        #     mySentinel = Sentinel(sentinel_list)
        #     redis_conn = mySentinel.master_for(configs.get('master_name'), password=configs.get('redis_password'))
        # else:
        #     redis_conn = redis.Redis(host=configs.get('redis_host'), port=configs.get('redis_port'), password=configs.get('redis_password'))
        if sentinel_list:
            mySentinel = Sentinel(sentinel_list)
            redis_conn = mySentinel.master_for(configs.get('master_name'), password=configs.get('redis_password'))
        else:
            redis_conn = redis.Redis(host=configs.get('redis_host'), port=configs.get('redis_port'),
                                     password=configs.get('redis_password'))
    except Exception as e:
        log.error("Connect Error!!!!", e)
        return None
    return redis_conn


def update_custom_user_info(group_id, tag, wait_group_id, my_tag, status):
    conn = pymysql.connect(host=configs.get('mycat_host'), port=configs.get('mycat_port'),
                           user=configs.get('mycat_usr'),
                           passwd=configs.get('mycat_passwd'), db=configs.get('mycat_db'))
    cur = conn.cursor()
    sql = "update custom_user_group set group_condition = '" + wait_group_id + ";" + tag + ";" + my_tag + ";" + status + "' "
    sql = sql + "where group_id = " + group_id
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()


def select_custom_user_info(group_id):
    conn = pymysql.connect(host=configs.get('mycat_host'), port=configs.get('mycat_port'),
                           user=configs.get('mycat_usr'),
                           passwd=configs.get('mycat_passwd'), db=configs.get('mycat_db'))
    cur = conn.cursor()
    sql = "select group_condition from custom_user_group where group_id = " + group_id
    df = pd.read_sql(sql, conn)
    df = pd.DataFrame(df, dtype=np.str).get_values()[0][0]
    cur.close()
    conn.close()
    return df


def connect_es_col(body, columns):
    df = connect_es(body)
    return df[columns]

# 查询ES数据，只保留标签数据，转化成DataFrame格式
def connect_es(body):
    res = get_es_data(body)
    # 将res转换成df二维表数据结构
    pandas_df = Select.from_dict(res).to_pandas()
    # 去除_score, _id, _type, _index列
    pandas_df.drop(['_score', '_id', '_type', '_index'], axis=1, inplace=True)
    log.info("pandas_df: \n{}".format(pandas_df))
    return pandas_df


# body: ES查询条件
# return: 查询结果
def get_es_data(body):
    es = get_es_connect()
    res = es.search(index=configs.get("es_index"), body=body)
    return res


def connect_es_scroll(body):
    return_df = pd.DataFrame()
    res = scroll_es_data(body, None)
    scroll_id = res.get('_scroll_id')
    pandas_df = Select.from_dict(res).to_pandas()
    pandas_df.drop(['_score', '_id', '_type', '_index'], axis=1, inplace=True)
    return_df = return_df.append(pandas_df)
    while scroll_id and len(res.get('hits').get('hits')) > 0:
        res = scroll_es_data(None, scroll_id)
        if len(res.get('hits').get('hits')) > 0:
            pandas_df = Select.from_dict(res).to_pandas()
            pandas_df.drop(['_score', '_id', '_type', '_index'], axis=1, inplace=True)
            return_df = return_df.append(pandas_df)
            scroll_id = res.get('_scroll_id')
    return_df = return_df.reset_index(drop=True)
    return return_df


def scroll_es_data(body, scroll_id):
    es = get_es_connect()
    if scroll_id is None:
        res = es.search(index=configs.get("es_index"), body=body, scroll="10m", size=configs.get("page_size"))
        return res
    else:
        res = es.scroll(scroll_id=scroll_id, scroll="10m")
        return res


def get_es_connect():
    if configs.get('es_https_status') == 1:
        # https,加密的
        es = Elasticsearch(
            configs.get("es_hosts"),
            verify_certs=False,
            http_auth=(configs.get("es_user"), configs.get("es_password")),
            # 在做任何操作之前，先进行嗅探
            sniff_on_start=configs.get('es_sniff_on_start'),
            # 节点没有响应时，进行刷新，重新连接
            sniff_on_connection_fail=True,
            use_ssl=True,
            # 每 60 秒刷新一次
            sniffer_timeout=configs.get("es_timeout"),
            es_timeout=configs.get("es_timeout"),
            es_user=configs.get("es_user"),
            es_password=configs.get("es_password")
        )
    else:
        es = Elasticsearch(
            configs.get("es_hosts"),
            http_auth=(configs.get("es_user"), configs.get("es_password")),
            # 在做任何操作之前，先进行嗅探
            sniff_on_start=configs.get('es_sniff_on_start'),
            # 节点没有响应时，进行刷新，重新连接
            sniff_on_connection_fail=True,
            # 每 60 秒刷新一次
            sniffer_timeout=configs.get("es_timeout"),
            es_timeout=configs.get("es_timeout"),
            es_user=configs.get("es_user"),
            es_password=configs.get("es_password")
        )
    return es


def get_field_type_mappings():
    es = get_es_connect()
    index_name = configs.get("es_index")
    response = es.indices.get_mapping(index=index_name)
    field_type_mappings = {}
    properties = response[index_name]['mappings']['properties']
    for k in properties:
        t = properties[k]['type']
        field_type_mappings[k] = t
    return field_type_mappings


# 获取所有字段类型为text的字段英文名
def get_text_fields():
    text_fields = []
    m = get_field_type_mappings()
    for k in m:
        v = m[k]
        if (v == 'text'):
            text_fields.append(k)
    return text_fields
