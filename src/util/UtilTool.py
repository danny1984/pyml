# coding: utf-8

import logging

logger = logging.getLogger('sgml_logger')
logger.setLevel(logging.DEBUG)

# 写入日志文件
fh = logging.FileHandler('sgml.log')
fh.setLevel(logging.INFO)
# 写入到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s-%(filename)s-%(lineno)d:    %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)