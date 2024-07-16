from config import Config
from loguru import logger
from elasticsearch import Elasticsearch
client = Elasticsearch(hosts=Config.ES_HOSTS)

all_indices = client.indices.get_alias()

# 遍历并删除每个索引
for index_name in all_indices:
    # 删除索引
    response = client.indices.delete(index=index_name)
    logger.info(f"Deleted index: {index_name}")