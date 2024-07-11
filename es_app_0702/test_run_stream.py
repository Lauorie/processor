# import os
# import re
# import json
# from loguru import logger
# from config import Config
# from flask_cors import CORS
# from esearch import Indexer
# from functools import lru_cache
# from rerank import Reranker, RougeReranker
# from vllm_llm import LLM
# from typing import List, Tuple, Dict
# from flask import Flask, Response, request, jsonify, stream_with_context
# from elasticsearch import ConnectionError, NotFoundError


# class HybridSearchService:
#     def __init__(self):
#         self.indexer = Indexer()
#         self.llm = LLM()
#         self.reranker = Reranker()
#         self.reranker_threshold = 0.5
        
#     def es_search(self, user_id: str, query_str: str, index_type: str, filenames: List[str] = None):
#         try:
#             return self.indexer.search_with_filename(user_id, index_type, query_str, filenames), "success"
#         except ConnectionError as e:
#             logger.error(f"Elasticsearch connection failed: {e}")
#             return [None] * 15, "es_search_failed"
#         except NotFoundError as e:
#             logger.error(f"Elasticsearch index not found: {e}")
#             return [None] * 15, "es_index_not_found"
        
#     def handle_search(self, user_id: str, query: str, index_type: str, search_more: int = 0, filenames: List[str] = None):         
#         try:
#             query_group = self.llm.generate_queries(query)
#         except Exception as e:
#             logger.error(f"vLLM generation failed: {str(e)}")
#             return None, None, "vllm_generation_failed"
        
#         results = []
#         for q in query_group:
#             if q:
#                 result, status = self.es_search(user_id, q, index_type, filenames)
#                 if status != "success": # 检查所有状态，而不仅是es_search_failed
#                     return None, None, status 
#                 results.append(result)
        
#         if not results:
#             return None, None, "empty_database"

#         recall_chunks = [chunk for result in results for chunk in result[0]]
#         recall_chunks = list(set(recall_chunks))
#         logger.info(f"针对问题:{query} 去重后的Recall chunks length: {len(recall_chunks)}")
#         logger.info(f"针对问题:{query} 去重后的Recall chunks: {recall_chunks}")
        
#         try:
#             rerank_chunks = self.reranker.rerank(query, [recall_chunks], k=len(recall_chunks), threshold=self.reranker_threshold)
#             logger.info(f"针对问题:{query} Rerank chunks length: {len(rerank_chunks)}")
#             rerank_chunks = rerank_chunks[:15] # 最多返回15个chunk,不然要OOM
#         except Exception as e:
#             logger.error(f"Rerank model failed: {str(e)}")
#             return None, None, "rerank_failed"
        
#         if not rerank_chunks:
#             logger.info(f"问题：{query} 在阈值为{self.reranker_threshold}时经过rerank后的chunks为空")
#             return None, None, "empty_after_rerank"
                
#         end_index = min((search_more + 1) * 5, len(rerank_chunks))
#         selected_rerank_chunks = rerank_chunks[:end_index]
#         logger.info(f"用户{user_id}的问题{query} 此次选择了Rerank中的第1到{end_index}的chunk进行RAG生成")
#         logger.info(f"针对问题:{query} All Rerank results length: {len(rerank_chunks)}")
#         logger.info(f"针对问题:{query} Selected Rerank results length: {len(selected_rerank_chunks)}")
#         logger.info(f"针对问题:{query} Selected Rerank results: {selected_rerank_chunks}")     
        
#         final_result = []
#         seen_chunks = {}

#         for chunk in selected_rerank_chunks:            
#             for result in results:  
#                 chunks = result[0]
#                 if chunk in chunks:
#                     index = chunks.index(chunk)
#                     chunk_id = result[7][index]
#                     seen_chunks[chunk_id] = {
#                         "chunk_text": chunk,
#                         "file_texts_path": result[1][index],
#                         "file_name_md5_new": result[2][index],
#                         "file_real_name": result[3][index],
#                         "file_type": result[4][index],
#                         "positions": result[5][index],
#                         "page_no": result[6][index],
#                         "chunk_id": chunk_id,
#                         "mysql_id": result[8][index],
#                         "file_from_folder": result[9][index],
#                         "file_size": result[10][index],
#                         "file_num_chars": result[11][index],
#                         "file_upload_time": result[12][index],
#                         "chunk_text_list": result[13][index],
#                         "base64_list": result[14][index]
#                     }
#         final_result = list(seen_chunks.values())        
#         final_data = json.dumps(final_result, ensure_ascii=False, indent=4)
#         return selected_rerank_chunks, final_data, "success"
    
#     def handle_no_streaming(self, query, rerank_chunks):
#         try:
#             response = self.llm.get_answer_from_pdf(rerank_chunks,query)
#             return response
#         except KeyboardInterrupt:
#             return '[WARNING] Generation interrupted'
    
#     def handle_streaming(self, query, rerank_chunks, history=None):
#         @stream_with_context
#         def generate():
#             try:
#                 for new_text in self.llm._chat_stream(query, rerank_chunks):
#                     yield new_text
#             except KeyboardInterrupt:
#                 yield '[WARNING] Generation interrupted'    
#         return Response(generate(), mimetype='text/event-stream')

#     def handle_json(self, user_id, query, index_type, search_more=0, history=None, file_names=None):
#         try:
#             selected_rerank_chunks, final_data, status = self.handle_search(user_id, query, index_type, search_more, file_names)
#             if status != "success":
#                 return handle_error(status) # 使用 handle_error 处理所有错误状态
            
#             logger.info("Returning search results")
#             return Response(final_data, mimetype='application/json')
            
#         except Exception as e:
#             logger.error(f"Error in handle_json: {e}")
#             return jsonify({"error": "An error occurred while processing the request"}), 500


# # 初始化
# def create_app():
#     app = Flask(__name__)
#     CORS(app)
#     app.config.from_object(Config)
#     return app

# app = create_app()

# # 初始化服务
# service = HybridSearchService()

# # 错误处理函数
# def handle_error(status):
#     error_messages = {
#         "vllm_generation_failed": "vLLM is down. Please try again or contact support.",
#         "es_search_failed": "Elasticsearch connection failed. Please check the connection and try again.",
#         "es_index_not_found": "Elasticsearch index not found. Please upload new documents and try again.", # 修改错误信息
#         "empty_database": "No related information found in the database. Please upload new documents.",
#         "rerank_failed": "Reranking process failed. Please try again later.",
#         "empty_after_rerank": "No results met the relevance threshold after reranking. Please try a different query.",
#     }
#     return jsonify({"error": error_messages.get(status, "An unknown error occurred")}), 500

# # 搜索和流处理函数
# def handle_search_and_streaming(user_id, query, index_type, search_more, history, filenames=None):
#     try:
#         selected_rerank_chunks, final_data, status = service.handle_search(user_id, query, index_type, search_more, filenames)
#         if status != "success":
#             return handle_error(status) # 使用 handle_error 处理所有错误状态

#         # 如果有图片返回，不能再回答 "无答案。"
#         answer = service.handle_no_streaming(query, selected_rerank_chunks)
#         json_data = json.loads(final_data)
        
#         # 全组
#         base_64_list = []
#         for i in json_data:
#             if isinstance(i["base64_list"], list):
#                 base_64_list.extend(i["base64_list"])
#             else:
#                 base_64_list.append(i["base64_list"])
        
#         # 只取第一组
#         base_64_list = []
#         if isinstance(json_data[0]["base64_list"], list):
#             base_64_list.extend(json_data[0]["base64_list"])
#         else:
#             base_64_list.append(json_data[0]["base64_list"])

#         base_64_list_setted = list(set(base_64_list))
#         if len(base_64_list_setted) > 1 and answer == "无答案。":
#             logger.info(f"图片数量：{len(base_64_list_setted)}")
#             return "如图："

#         return service.handle_streaming(query, selected_rerank_chunks, history)
#     except Exception as e:
#         app.logger.error(f"Error in handle_search_and_streaming: {str(e)}")
#         return jsonify({"error": "An error occurred while processing the request"}), 500

# # 路由定义
# @app.route("/stream/V3/", methods=['POST'])
# def stream_endpoint():
#     data = request.get_json(silent=True)
#     if not data:
#         return jsonify({"error": "Invalid or missing JSON"}), 400

#     query = data.get('query')
#     user_id = data.get('user_id')
#     index_type = data.get('index_type', 'knowledge_base_v1')
#     search_more = data.get('search_more', 0)
#     history = data.get('history', [])
#     file_names = data.get('file_names', [])
    
#     if not query or not user_id or not index_type:
#         return jsonify({"error": "Missing query, user_id or index_type"}), 400

#     # 第一层判断query和历史是否有联系，如果有联系则走模型，如果没有则走RAG
#     # 第二层如果提供了file_names则走知识库knowledge_base，如果没有则走长文档问答long_context
#     # 前端传参时，知识库的file_names是一个空列表，长文档问答的file_names是List[str]
#     if history:
#         router_answer = service.llm.re_router(query)
#         if router_answer == 'rag':
#             return handle_search_and_streaming(user_id, query, index_type, search_more, history, file_names)
#         elif router_answer == "translation":
#             return service.llm.get_translation_answer(history)
#         elif router_answer == "summary":
#             return service.llm.get_summary_answer(history)
#         elif router_answer == "table":
#             return service.llm.get_table_answer(history)
#     else:
#         return handle_search_and_streaming(user_id, query, index_type, search_more, history, file_names)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5069, debug=True)

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, 'summit define')
]
# No need to add instruction for retrieval documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
input_texts = queries + documents
model_path = '/root/web_demo/HybirdSearch/models/models--liuqi6777--pe-rank-mistral-jina'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

max_length = 8192

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
outputs = model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
