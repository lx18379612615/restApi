from Logger import Logger

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from connect_unit import get_text_fields
log = Logger(logname="app.log", logger="restApi").getlog()


def clean_x_datas(df):
    # 去除NAN数据，axis=0表示行，1表示列；how='any'表示一行中只要有一列的值是NAN，就去除这一行；inplace=True表示直接在元数据上进行修改；
    df.dropna(axis=0, how='any', inplace=True)
    # 重置df的索引
    df = df.reset_index(drop=True)
    cols = df.columns
    text_fields = list(set(get_text_fields()).intersection(set(cols)))
    if len(text_fields) > 0:
        # text_df = df[text_fields]
        text_df = df
        text_df.astype('str')
        log.info("text_df: ")
        log.info(text_df)
        enc = OrdinalEncoder()
        enc.fit(text_df)
        text_df = pd.DataFrame(enc.transform(text_df))
        text_df.columns = text_fields
        log.info(text_df)
        for field in text_fields:
            df[field] = text_df[field]
    df = pd.DataFrame(format_feature(df))
    df = df.astype('float64')
    df.columns = cols
    log.info(df)
    return df

def format_feature(x_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    return x_train

# 数据转换
# 入参：df: 所有种子用户[tag]标签数据; positive_value种子客户判别规则
# 出参：df: 转换之后的标签数据; transformed_positive_value: transformed_positive_value
def clean_y_datas(df, positive_value):
    df.fillna('0', inplace=True)
    # 以下三行是做数据转换，将离散型的数据转换成 0 到 n − 1 之间的数，这里 n 是一个列表的不同取值的个数，可以认为是某个特征的所有不同取值的个数
    le = LabelEncoder()
    le.fit(df)
    df = pd.DataFrame(le.transform(df))
    # 特殊处理
    if len(le.classes_) == 1:
        # 都是同样的数据
        transformed_positive_value = le.transform([le.classes_[0]])[0]
    else:
        transformed_positive_value = le.transform([positive_value])[0]
    return df, transformed_positive_value
