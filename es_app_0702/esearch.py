from typing import List
from loguru import logger
from config import Config
# from bm25 import BM25Ranker_attach
from elasticsearch import Elasticsearch
from embeddings import EmbeddingsClient


class Indexer:
    def __init__(self):
        self.client = Elasticsearch(hosts=Config.ES_HOSTS)
        self.model = EmbeddingsClient()
        # self.bm25_ranker = BM25Ranker_attach()

    def search_(self, user_id, index_type, statement):
        index_name = f"{index_type}_{user_id}"
        
        match_response = self.client.search(
            index=index_name,
            size=5,
            query={"match":{"file_docai_json.text": statement}}   # 搜索的区域记得改！！！
        )
        
        knn_response = self.client.search(
            index=index_name,
            size=5,
            knn={
                "field": "file_docai_json.text_embeddings",   # 搜索的区域记得改！！！
                "query_vector": self.model.encode(statement).tolist(),  
                "k": 5,
                "num_candidates": 10,
            }
        )
        
        match_docs = match_response['hits']['hits'] 
        knn_docs = knn_response['hits']['hits']
        merged_docs = match_docs + knn_docs
        
        chunk_text = [i['_source']['file_docai_json']['text'] for i in merged_docs]
        file_texts_path = [i['_source']['file_texts_path'] for i in merged_docs]  # 这个路径是文本文件的路径
        file_name_md5_new = [i['_source']['file_name_md5_new'] for i in merged_docs]
        file_real_name = [i['_source']['file_real_name'] for i in merged_docs]
        file_type = [i['_source']['file_type'] for i in merged_docs]
        positions = [i['_source']['file_docai_json']['positions'] for i in merged_docs]  
        page_no = [i['_source']['file_docai_json']['page_no'] for i in merged_docs]
        chunk_id = [i['_source']['file_docai_json']['id'] for i in merged_docs]
        mysql_id = [i['_source']['mysql_id'] for i in merged_docs]
        file_from_folder = [i['_source']['file_from_folder'] for i in merged_docs]
        file_size = [i['_source']['file_size'] for i in merged_docs]
        file_num_chars = [i['_source']['file_num_chars'] for i in merged_docs]
        file_upload_time = [i['_source']['file_upload_time'] for i in merged_docs]
        chunk_text_list = [i['_source']['file_docai_json']['text_list'] for i in merged_docs]
        base64_list = [i['_source']['file_docai_json']['base64_list'] for i in merged_docs]

        return chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, chunk_text_list, base64_list
    
    def search_with_filename(self, user_id, index_type, statement, file_names:List[str]=None):
        """
        如果指定了file_names参数，则只搜索指定文件名内的文档。
        如果未指定file_names参数，则搜索所有文档。
        """
        index_name = f"{index_type}_{user_id}"
        
        # 构建基本查询
        match_query = {
            "bool": {
                "must": [
                    {"match": {"file_docai_json.text": statement}}
                ]
            }
        }
        
        knn_query = {
            "field": "file_docai_json.text_embeddings",
            "query_vector": self.model.encode(statement).tolist(),
            "k": 5,
            "num_candidates": 10
        }
        
        all_filename_in_es = self.find_all_filename_from_es(index_name)
        
        if file_names:
            logger.info(f"用户{user_id}的问题{statement}指定了文件名称: {file_names}")
            if set(file_names).issubset(all_filename_in_es):
                logger.info(f"指定的文件名称：{file_names}")
                match_query["bool"]["filter"] = [
                    {"terms": {"file_name_md5_new": file_names}}
                ]
                knn_query["filter"] = {"terms": {"file_name_md5_new": file_names}}
            else:
                missing_files = set(file_names) - set(all_filename_in_es)
                logger.warning(f"以下文件未在ES中检索到：{missing_files}，请重新上传。")
                return  # 如果有文件未找到，提前返回，不执行后续操作
        else:
            logger.info(f"用户{user_id}的问题{statement}   未指定文件名称，搜索所有文件。")
        
        # 使用BM25进行文本匹配
        match_response = self.client.search(
            index=index_name,
            size=5,
            query=match_query
        )
        
        # 使用KNN进行文本匹配
        knn_response = self.client.search(
            index=index_name,
            size=5,
            knn=knn_query
        )
        
        # 合并两个搜索结果
        match_docs = match_response['hits']['hits'] 
        knn_docs = knn_response['hits']['hits']
        merged_docs = match_docs + knn_docs

        # 获取召回的chunks、文本等信息
        chunk_text = [i['_source']['file_docai_json']['text'] for i in merged_docs]
        file_texts_path = [i['_source']['file_texts_path'] for i in merged_docs]  # 这个路径是文本文件的路径
        file_name_md5_new = [i['_source']['file_name_md5_new'] for i in merged_docs]
        file_real_name = [i['_source']['file_real_name'] for i in merged_docs]
        file_type = [i['_source']['file_type'] for i in merged_docs]
        positions = [i['_source']['file_docai_json']['positions'] for i in merged_docs]  
        page_no = [i['_source']['file_docai_json']['page_no'] for i in merged_docs]
        chunk_id = [i['_source']['file_docai_json']['id'] for i in merged_docs]
        mysql_id = [i['_source']['mysql_id'] for i in merged_docs]
        file_from_folder = [i['_source']['file_from_folder'] for i in merged_docs]
        file_size = [i['_source']['file_size'] for i in merged_docs]
        file_num_chars = [i['_source']['file_num_chars'] for i in merged_docs]
        file_upload_time = [i['_source']['file_upload_time'] for i in merged_docs]
        text_list = [i['_source']['file_docai_json']['text_list'] for i in merged_docs]
        base64_list = [i['_source']['file_docai_json']['base64_list'] for i in merged_docs]

        return chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, text_list, base64_list
    
    # 查看index_name内的所有文件名
    def find_all_filename_from_es(self, index_name):        
        query_ = {
            "size": 0,
            "aggs": {
                "file_names": {
                    "terms": {
                        "field": "file_name_md5_new",
                        "size": 10000000  # 设置一个足够大的值以确保返回所有文件名
                    }
                }
            }
        }
        try:
            result_ = self.client.search(index=index_name, body=query_)
            all_filename_in_es = [bucket['key'] for bucket in result_['aggregations']['file_names']['buckets']]
            return all_filename_in_es
        except Exception as e:
            logger.error(f"查询所有文件名时出错：{e}")
            return []
    
    # 辅助main
    def remove_duplicates(self,data):
        seen = set()
        result = []
        for item in data:
            file_name_md5 = item['_source']['file_name_md5_new']
            if file_name_md5 not in seen:
                seen.add(file_name_md5)
                result.append(item)
        return result
    
    
    # main
    # def search_chunk_by_query_text(self, user_id, index_type, statement, file_names=None):
    def search_chunk_by_query_text(self, user_id, index_type, statement, file_names:List[str]=None):
        """
        如果指定了file_names参数，则只搜索指定文件名内的文档。
        如果未指定file_names参数，则搜索所有文档。
        """
        index_name = f"{index_type}_{user_id}"
        
        # 构建基本查询
        match_query = {
            "bool": {
                "must": [
                    {"match": {"file_docai_json.text": statement}}
                ]
            }
        }
        
        knn_query = {
            "field": "file_docai_json.text_embeddings",
            "query_vector": self.model.encode(statement).tolist(),
            "k": 5,
            "num_candidates": 10
        }
        
        all_filename_in_es = self.find_all_filename_from_es(index_name)
        
        if file_names:
            logger.info(f"用户{user_id}的问题{statement}指定了文件名称: {file_names}")
            if set(file_names).issubset(all_filename_in_es):
                logger.info(f"指定的文件名称：{file_names}")
                match_query["bool"]["filter"] = [
                    {"terms": {"file_name_md5_new": file_names}}
                ]
                knn_query["filter"] = {"terms": {"file_name_md5_new": file_names}}
            else:
                missing_files = set(file_names) - set(all_filename_in_es)
                logger.warning(f"以下文件未在ES中检索到：{missing_files}，请重新上传。")
                return  # 如果有文件未找到，提前返回，不执行后续操作
        else:
            logger.info(f"用户{user_id}的问题{statement}   未指定文件名称，搜索所有文件。")
        
        # 使用BM25进行文本匹配
        match_response = self.client.search(
            index=index_name,
            size=5,
            query=match_query
        )
        
        # 使用KNN进行文本匹配
        knn_response = self.client.search(
            index=index_name,
            size=5,
            knn=knn_query
        )
        
        # 合并两个搜索结果
        match_docs = match_response['hits']['hits'] 
        knn_docs = knn_response['hits']['hits']
        merged_docs = match_docs + knn_docs

        return merged_docs
    
    
    # def search_chunk_by_query_attach(self, user_id, index_type, statement,file_names=None):
    #     """
    #     返回一个列表，后续还需要通过[序号]['_source']['file_docai_json']得到详细的属性,
    #     并且unique_chunks可能来自不同的文件
    #     """
    #     index_name = f"{index_type}_{user_id}"
        
    #     all_filename_in_es = self.find_all_filename_from_es(index_name)
    #     match_query = {
    #         "bool": {
    #             "must": [
    #                 {"match": {"file_docai_json.attach_text": statement}}
    #             ]
    #         }
    #     }
    #     if file_names:
    #         logger.info(f"用户{user_id}的问题{statement}指定了文件名称: {file_names}")
    #         if set(file_names).issubset(all_filename_in_es):
    #             logger.info(f"指定的文件名称：{file_names}")
    #             match_query["bool"]["filter"] = [
    #                 {"terms": {"file_name_md5_new": file_names}}
    #             ]
    #         else:
    #             missing_files = set(file_names) - set(all_filename_in_es)
    #             logger.warning(f"以下文件未在ES中检索到：{missing_files}，请重新上传。")
    #             return  # 如果有文件未找到，提前返回，不执行后续操作
    #     else:
    #         logger.info(f"用户{user_id}的问题{statement}   未指定文件名称，搜索所有文件。")
        
    #     response = self.client.search(
    #         index=index_name,
    #         size=50,
    #         query=match_query
    #     )
    #     docs = response['hits']['hits']
        
    #     sorted_docs = self.bm25_ranker.rank(docs, statement)
    #     return sorted_docs
    def search_chunk_by_query_attach(self, user_id, index_type, statement, file_names:List[str]=None):
        """
        如果指定了file_names参数，则只搜索指定文件名内的文档。
        如果未指定file_names参数，则搜索所有文档。
        """
        index_name = f"{index_type}_{user_id}"
        
        # 构建基本查询
        match_query = {
            "bool": {
                "must": [
                    {"match": {"file_docai_json.attach_text": statement}}
                ]
            }
        }
        
        knn_query = {
            "field": "file_docai_json.text_embeddings",
            "query_vector": self.model.encode(statement).tolist(),
            "k": 5,
            "num_candidates": 10
        }
        
        all_filename_in_es = self.find_all_filename_from_es(index_name)
        
        if file_names:
            logger.info(f"用户{user_id}的问题{statement}指定了文件名称: {file_names}")
            if set(file_names).issubset(all_filename_in_es):
                logger.info(f"指定的文件名称：{file_names}")
                match_query["bool"]["filter"] = [
                    {"terms": {"file_name_md5_new": file_names}}
                ]
                knn_query["filter"] = {"terms": {"file_name_md5_new": file_names}}
            else:
                missing_files = set(file_names) - set(all_filename_in_es)
                logger.warning(f"以下文件未在ES中检索到：{missing_files}，请重新上传。")
                return  # 如果有文件未找到，提前返回，不执行后续操作
        else:
            logger.info(f"用户{user_id}的问题{statement}   未指定文件名称，搜索所有文件。")
        
        # 使用BM25进行文本匹配
        match_response = self.client.search(
            index=index_name,
            size=5,
            query=match_query
        )
        
        # 使用KNN进行文本匹配
        knn_response = self.client.search(
            index=index_name,
            size=5,
            knn=knn_query
        )
        
        # 合并两个搜索结果
        match_docs = match_response['hits']['hits'] 
        knn_docs = knn_response['hits']['hits']
        merged_docs = match_docs + knn_docs

        return merged_docs
    
    def search_chunk_by_title_filename(self, user_id, index_type, file_name_md5):
        index_name = f"{index_type}_{user_id}"
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"user_id": user_id}},
                        {"match": {"file_name_md5_new": file_name_md5}},
                        {"match": {"file_docai_json.type": 'TITLE'}}
                        ]
                    }
                }
            }       
        match = self.client.search(
            index=index_name,
            body=query,
            size=1000
        )
        
        docs = match['hits']['hits']
        
        return docs
    
    def search_chunk_by_section_title_filename(self, user_id, index_type, file_name_md5):
        index_name = f"{index_type}_{user_id}"
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"user_id": user_id}},
                        {"match": {"file_name_md5_new": file_name_md5}},
                        {"match": {"file_docai_json.type": 'SECTION-TITLE'}}
                        ]
                    }
                }
            }       
        match = self.client.search(
            index=index_name,
            body=query,
            size=1000
        )
        
        docs = match['hits']['hits']
        
        return docs
    
    def serach_chunk_by_file_name_and_parentid(self, user_id, index_type, file_name_md5, parent_id):
        index_name = f"{index_type}_{user_id}"
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"user_id": user_id}},
                        {"match": {"file_name_md5_new": file_name_md5}},
                        {"match": {"file_docai_json.parent_id": parent_id}}
                        ]
                    }
                }
            }       
        match = self.client.search(
            index=index_name,
            body=query,
            size=10000
        )
        
        docs = match['hits']['hits']
        return docs
    
    # 新增2 依据文件名和页码进行搜索
    def serach_chunk_by_file_name_and_page(self, user_id, index_type, file_name_md5, page_no):
        index_name = f"{index_type}_{user_id}"
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"user_id": user_id}},
                        {"match": {"file_name_md5_new": file_name_md5}},
                        {"match": {"file_docai_json.page_no": page_no}}
                        ]
                    } 
                }
            }       
        match = self.client.search(index=index_name, body=query, size=1500)
        docs = match['hits']['hits']
        
        return docs
    
    # 新增2 依据文件名进行搜索
    def serach_chunk_by_file_name(self, user_id, index_type, file_name_md5):
        index_name = f"{index_type}_{user_id}"
        # test
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"user_id": user_id}},
                        {"match": {"file_name_md5_new": file_name_md5}}
                        ]
                    } 
                }
            }       
        match = self.client.search(index=index_name,body=query,size=10000)
        docs = match['hits']['hits']
        return docs