import gc
import torch
import logging
from tqdm import tqdm
from loguru import logger
from config import Config
from elasticsearch import Elasticsearch
from embeddings import EmbeddingsClient
from flask import Flask, request, jsonify
from sentence_transformers import LoggingHandler, SentenceTransformer
from pre_process_merge_re import KnowledgeDocumentPreprocessor


logger.add(Config.UPLOAD_LOG_PATH)
app = Flask(__name__)

class DocumentIndexer:
    def __init__(self, es_hosts):       
        self.client = Elasticsearch(hosts=es_hosts)
        self.model = EmbeddingsClient()

    def process_json(self, 
                     processed_data, # DOCAI解析完的json文件再经过处理后的数据
                     user_id, 
                     mysql_id, 
                     file_json_path, 
                     file_name_md5_new, 
                     file_real_name, 
                     file_type, 
                     file_used_for, 
                     file_texts_path,
                     file_from_folder,
                     file_size,
                     file_num_chars,
                     file_upload_time):
        index_name = f"{file_used_for}_{user_id}" 
        
        ES_BATCH_SIZE = 32
        EMBEDDING_BATCH_SIZE = 256
        
        if file_used_for == 'long_context_v1':
            text_list = [str(doc['attach_text'])[:1024] for doc in processed_data if doc.get('attach_text')] # 硬取前1024个字符，不然超长的文本会导致OOM
        else:
            text_list = [str(doc['text'])[:1024] for doc in processed_data if doc.get('text')]
        all_embeddings = []
        try:
            for start_idx in tqdm(range(0, len(text_list), EMBEDDING_BATCH_SIZE), desc="Encoding texts"):
                end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, len(text_list))
                batch_texts = text_list[start_idx:end_idx]
                batch_embeddings = self.model.encode(batch_texts)
                all_embeddings.extend(batch_embeddings.tolist())
        
        
          
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error("RuntimeError: CUDA error: out of memory during encoding")
                raise RuntimeError("CUDA error: out of memory during encoding")
            else:
                raise e

        embedding_idx = 0
        for doc in processed_data:
            if doc.get('text'):
                doc['text_embeddings'] = all_embeddings[embedding_idx]
                embedding_idx += 1
            else:
                doc['text_embeddings'] = None
            
        operations = []
        for i, data in tqdm(enumerate(processed_data, 1), desc="Indexing documents"):
            operation = {
                "index": {
                    "_index": index_name,
                    "_id": f"{user_id}_{mysql_id}_{file_name_md5_new}_{i}"
                }
            }
            formatted_data = {
                "user_id": user_id,
                "mysql_id": mysql_id,
                "file_json_path": file_json_path,
                "file_name_md5_new": file_name_md5_new,
                "file_real_name": file_real_name,
                "file_type": file_type,
                "file_used_for": file_used_for,
                "file_texts_path": file_texts_path,
                "file_from_folder": file_from_folder,
                "file_size": file_size,
                "file_num_chars": file_num_chars,
                "file_upload_time": file_upload_time,         
                "file_docai_json": data
            }
            operations.extend([operation, formatted_data])
            
            if i % (ES_BATCH_SIZE * 2) == 0 or i == len(processed_data):
                self.index_documents(index_name, operations)
                operations = []
                
        self.wipe_cache()
        logger.info(f"已处理并索引 {len(processed_data)} 个chunk")
 
    def wipe_cache(self): 
        gc.collect()
        torch.cuda.empty_cache()
        
    def index_documents(self, index_name, operations):
        self.setup_mapping(index_name)
        response = self.client.bulk(index=index_name, operations=operations, refresh=True)
        logger.info(f"response: {response}")
        return response
            
    def setup_mapping(self, index_name):
        index_mapping = {
            "mappings": {
                # "dynamic": "strict", # 严格模式，不允许动态添加字段
                "properties": {
                    "user_id": {"type": "keyword"},
                    "mysql_id": {"type": "keyword"},
                    "file_json_path": {"type": "text"},
                    "file_name_md5_new": {"type": "keyword"},
                    "file_real_name": {"type": "text"},
                    "file_type": {"type": "keyword"},
                    "file_used_for": {"type": "keyword"},
                    "file_texts_path": {"type": "keyword"},
                    "file_from_folder": {"type": "keyword"},
                    "file_size": {"type": "keyword"},
                    "file_num_chars": {"type": "keyword"},
                    "file_upload_time": {"type": "keyword"},
                    "file_docai_json": {
                        "properties": {
                            "display": {
                                "type": "nested",
                                "properties": { 
                                    "left": {"type": "float"},
                                    "top": {"type": "float"},
                                    "right": {"type": "float"},
                                    "bottom": {"type": "float"},
                                    "page_width": {"type": "float"},
                                    "page_height": {"type": "float"}
                                }
                            },
                            "positions": {
                                "type": "nested",
                                "properties": {
                                    "bbox": {"type": "float"},
                                    "page_height": {"type": "float"},
                                    "page_no": {"type": "integer"},
                                    "page_width": {"type": "float"}
                                }
                            },
                            "page_no": {"type": "integer"},
                            "text": {"type": "text"},
                            "text_embeddings": {
                                "type": "dense_vector",
                                "dims": 1792,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "text_list": {"type": "text"},   # 6/19新增
                            "base64_list": {"type": "text"}, # 6/19新增
                            "type": {"type": "keyword"},
                            "attach_text": {"type": "text"},
                            "id": {"type": "keyword"},
                            "parent_id": {"type": "keyword"}
                        }
                    }
                }
            }
        }



            # 检查索引是否存在，如果不存在，则创建新的索引
        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body=index_mapping)
            logger.info(f"已创建Elasticsearch索引: {index_name}")
        else:
            logger.info(f"索引 {index_name} 已存在，数据将被追加到该索引中")
    
    
    def search_children(self, user_id, index_type, query):
        index_name = f"{index_type}_{user_id}"       
        
        # Step 1: Perform KNN search to find the most similar document
        knn_response = self.client.search(
            index=index_name,
            size=1,
            knn={
                "field": "file_docai_json.text_embeddings",
                "query_vector": self.model.encode(query).tolist(),
                "k": 1,
                "num_candidates": 5,
            }
        )
        
        knn_docs = knn_response['hits']['hits']
        
        if not knn_docs:
            return []

        # Extract the parent_id of the most similar document
        parent_id = knn_docs[0]['_source']['file_docai_json']['parent_id']
        logger.info(f"Parent ID: {parent_id}")
        
        # Step 2: Search for all documents with the same parent_id
        search_response = self.client.search(
            index=index_name,
            size=100,
            query={"match":{ "file_docai_json.parent_id": parent_id }}   # 搜索的区域记得改！！！
        )
        
        # Extract texts from the search results and return them in order
        search_docs = search_response['hits']['hits']
        child_texts = [i['_source']['file_docai_json']['text'] for i in search_docs]
        file_texts_path = [i['_source']['file_texts_path'] for i in search_docs]
        positions = [i['_source']['file_docai_json']['positions'] for i in search_docs]
        page_no = [i['_source']['file_docai_json']['page_no'] for i in search_docs]        
        return child_texts


# 后端上传的参数        
REQUIRED_FIELDS = [
    'user_id', 'mysql_id', 'file_json_path', 'file_name_md5_new', 'file_real_name', 'file_type', 
    'file_used_for', 'file_texts_path', 'file_from_folder', 'file_size', 'file_num_chars', 
    'file_upload_time'
]

def validate_json(json_data):
    """验证请求中的JSON数据是否包含所有必需字段。"""
    if not json_data:
        return False, 'Missing JSON in request'
    if not all(field in json_data for field in REQUIRED_FIELDS):
        return False, 'Missing data in JSON'
    return True, None

@app.route('/upload/V2', methods=['POST'])
def upload():
    json_data = request.json
    is_valid, error_message = validate_json(json_data)
    if not is_valid:
        return jsonify({'error': error_message}), 400

    # 解包所有必需字段
    user_id = json_data['user_id']
    mysql_id = json_data['mysql_id']
    file_json_path = json_data['file_json_path']
    file_name_md5_new = json_data['file_name_md5_new']
    file_real_name = json_data['file_real_name']
    file_type = json_data['file_type']
    file_used_for = json_data['file_used_for']
    file_texts_path = json_data['file_texts_path']
    file_from_folder = json_data['file_from_folder']
    file_size = json_data['file_size']
    file_num_chars = json_data['file_num_chars']
    file_upload_time = json_data['file_upload_time']

    logger.info(f"收到来自用户ID：{user_id} 的文件{file_real_name}，用于{file_used_for}。")

    # 处理文件
    tokenizer_path = Config.MODEL_NAME_OR_PATH
    try:        
        dp = KnowledgeDocumentPreprocessor(tokenizer_path,  file_json_path, file_real_name) 
        knowledge_base_data, long_context_data = dp.process()
        
        # 创建ES索引
        es_hosts = Config.ES_HOSTS
        indexer = DocumentIndexer(es_hosts)

        if file_used_for == "long_context_v1":
            processed_data = long_context_data         
        else:
            processed_data = knowledge_base_data
            
        logger.info(f"完成用户ID：{user_id} 的文件{file_real_name}的处理。生成{len(processed_data)}个文本块。")  
        indexer.process_json(
                processed_data, user_id, mysql_id, file_json_path, file_name_md5_new, file_real_name, 
                file_type, file_used_for, file_texts_path, file_from_folder, file_size, 
                file_num_chars, file_upload_time) 

        return jsonify({'message': 'Data processed successfully'}), 200
    except RuntimeError as e:
        if 'CUDA error: out of memory' in str(e):
            logger.error("CUDA error: out of memory")
            return jsonify({'error': 'CUDA error: out of memory'}), 500
        else:
            logger.error(f"Unexpected error: {str(e)}")
            return jsonify({'error': 'An unexpected error occurred'}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.UPLOAD_PORT, debug=True)
