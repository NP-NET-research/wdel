#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @Author  :  xuzf
# @Contact :  xuzhengfei-email@qq.com
# @Create  :  2022-01-17 17:43:07
# @Update  :  2023-05-31 14:02:39
# @Desc    :  None
# =============================================================================

import os, sys
import logging
from time import strftime, localtime

LOG_FORMAT = "%(asctime)s - %(levelname)s (%(filename)s:%(lineno)s)\t%(message)s"
LOG_HOME = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'log')

def get_logger(logger_name, output_dir=None, stdout=True):
    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        log_file_name = "{}/{}.log".format(output_dir, logger_name + strftime(' %m-%d %H:%M', localtime()))
        fhandler = logging.FileHandler(log_file_name, mode="a", delay=False)
        handlers = [fhandler]
        if stdout:
            shandler = logging.StreamHandler(sys.stdout)
            handlers.append(shandler)
        logging.basicConfig(
            format=LOG_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=handlers
        )
    else:
        logging.basicConfig(
            format=LOG_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    return logger