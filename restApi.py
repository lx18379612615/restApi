# -*- coding: utf-8 -*-
from gevent import monkey

from Logger import Logger

monkey.patch_all()
from bottle import Bottle
import spread
import cluster
import config_default
import traceback
import json
import DataResponse

log = Logger(logname="app.log", logger="restApi").getlog()


configs = config_default.configs

app = Bottle()


@app.route('/get_main_feature', method='POST')
def get_main_feature():
    try:
        log.info("calling spread.get_main_feature() ...")
        return spread.get_main_feature()
    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        error_response = json.dumps(DataResponse.DataResponse('', 9000, e.__format__()).__dict__, ensure_ascii=False)
        return error_response

@app.route('/predict', method='POST')
def predict():
    try:
        log.info("calling cluster.predict()111 ...")
        return cluster.predict()
    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        error_response = json.dumps(DataResponse.DataResponse('', 9000, e.__format__()).__dict__, ensure_ascii=False)
        return error_response

@app.route('/population_spread', method='POST')
def population_spread():
    try:
        log.info("calling spread.population_spread() ...")
        return spread.population_spread()
    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        error_response = json.dumps(DataResponse.DataResponse('', 9000, e.__format__()).__dict__, ensure_ascii=False)
        return error_response

log.info("starting app with config " + json.dumps(configs) + "...")
app.run(host=configs.get('service_host'), port=configs.get('service_port'), server='gevent')
# run(host=configs.get('service_host'), port=configs.get('service_port'))
log.info("app stoped.")

