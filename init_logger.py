# -*- coding: utf-8 -*-
# @project : Denoise-tensorflow
# @Time : 2019-08-11 15:41 
# @Author : ZhangXiao(sinceresky@foxmail.com)

import logging
import os


def get_logger(_loggerDir, log_file, logger_name):
    if not os.path.exists(_loggerDir):
        os.mkdir(_loggerDir)
    _LogFile = os.path.join(_loggerDir, log_file)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # ccreate file handler which logs even debug messages
    fh = logging.FileHandler(_LogFile)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    _LogFormat = logging.Formatter("%(asctime)2s -%(name)-12s:  %(levelname)-10s - %(message)s")

    fh.setFormatter(_LogFormat)
    console.setFormatter(_LogFormat)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(console)
    return logger
