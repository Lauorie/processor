# import os
# import re
# import json
# from loguru import logger
# from config import Config
# from flask_cors import CORS
# from ESearch import Indexer
# from rerank import Reranker, RougeReranker
# from llm_vllm import LLMPredictor
# from typing import List, Tuple, Dict
# from flask import Flask, Response, request, jsonify, stream_with_context

# os.environ["CUDA_VISIBLE_DEVICES"] = Config.RUN_DEVICES

# class HybridSearchService:
#     def __init__(self, es_hosts, embedding_model_path, model_name_or_path, reranker_model_path):
#         self.indexer = Indexer(es_hosts, embedding_model_path)
#         self.llm_predictor = LLMPredictor(model_name_or_path)
#         self.reranker = Reranker(reranker_model_path)
        
#     def es_search(self, user_id: str, query_str: str, index_type: str, filenames: List[str] = None):
#         try:
#             return self.indexer.search_with_filename(user_id, index_type, query_str, filenames)
#         except Exception as e:
#             logger.error(f"Search failed: {e}")
#             return [None] * 15
        
#     def handle_search(self, user_id: str, query: str, index_type: str, search_more: int = 0, filenames: List[str] = None):         
#         query_group = self.llm_predictor.generate_queries(query)
#         results = [self.es_search(user_id, q, index_type, filenames) for q in query_group if q]
#         recall_chunks =[chunk for result in results for chunk in result[0]]
#         recall_chunks = list(set(recall_chunks))
#         logger.info(f"针对问题:{query} 去重后的Recall chunks length: {len(recall_chunks)}")
#         logger.info(f"针对问题:{query} 去重后的Recall chunks: {recall_chunks}")
        
#         # recall_chunks需要的格式是List[List[str]] 返回的是List[str]
#         rerank_chunks = self.reranker.rerank(query, [recall_chunks], k=len(recall_chunks))
#         logger.info(f"针对问题:{query} Rerank chunks length: {len(rerank_chunks)}")
#         rerank_chunks = rerank_chunks[:15] # 最多返回15个chunk,不然要OOM
        
#         if not rerank_chunks:
#             logger.info(f"问题：{query} 在阈值为0.3时经过rerank后的chunks为空")
#             return None, "No related information found, please upload new documents"
                
#         end_index = min((search_more + 1) * 5, len(rerank_chunks))
#         selected_rerank_chunks = rerank_chunks[:end_index]
#         logger.info(f"用户{user_id}的问题{query} 此次选择了Rerank中的第1到{end_index}的chunk进行RAG生成")
#         logger.info(f"针对问题:{query} All Rerank results length: {len(rerank_chunks)}")
#         logger.info(f"针对问题:{query} Selected Rerank results length: {len(selected_rerank_chunks)}")
#         logger.info(f"针对问题:{query} Selected Rerank results: {selected_rerank_chunks}")     
        
#         """chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, 
#         chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, chunk_text_list, base64_list """       
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
#         # 将去重后的结果转换回列表
#         final_result = list(seen_chunks.values())        
#         final_data = json.dumps(final_result, ensure_ascii=False, indent=4)
#         return selected_rerank_chunks, final_data
    
#     def handle_no_streaming(self, query, rerank_chunks):
#         try:
#             response = self.llm_predictor.get_answer_from_pdf(rerank_chunks,query)
#             return response
#         except KeyboardInterrupt:
#             return '[WARNING] Generation interrupted'
    
#     def handle_streaming(self, query, rerank_chunks, history=None):
#         @stream_with_context
#         def generate():
#             try:
#                 for new_text in self.llm_predictor._chat_stream(query, rerank_chunks):
#                     yield new_text
#             except KeyboardInterrupt:
#                 yield '[WARNING] Generation interrupted'    
#         return Response(generate(), mimetype='text/event-stream')

#     def handle_json(self, user_id, query, index_type, search_more=0, history=None, file_names=None):
#         try:
#             selected_rerank_chunks, final_data = self.handle_search(user_id, query, index_type, search_more, file_names)
#             # 如果 selected_rerank_chunks 为空，返回提示信息
#             if selected_rerank_chunks is None:
#                 error_message = "No related information found, please upload new documents"
#                 logger.info("No related document information found")
#                 return jsonify({"error": error_message}), 200

#             logger.info("Returning search results")
#             return Response(final_data, mimetype='application/json')
        
#         except Exception as e:
#             logger.error(f"Error in handle_json: {e}")
#             return jsonify({"error": "An error occurred while processing the request"}), 500
         
    
# app = Flask(__name__) 
# CORS(app)
# service = HybridSearchService(
#     es_hosts=Config.ES_HOSTS,
#     embedding_model_path=Config.EMBEDDING_MODEL_PATH,
#     model_name_or_path=Config.MODEL_PATH,
#     reranker_model_path=Config.RERANKER_MODEL_PATH
# )


# def handle_search_and_streaming(user_id, query, index_type, search_more, history, filenames=None):  
#     try:  
#         selected_rerank_chunks, _ = service.handle_search(user_id, query, index_type, search_more, filenames) 
#         if selected_rerank_chunks is None:
#             logger.info("No related document information found")
#             return jsonify({"error": "No related information found, please upload new documents"}), 500
        
#         return service.handle_streaming(query, selected_rerank_chunks, history)
#     except Exception as e:  
#         app.logger.error(f"vLLM generation failed: {str(e)}")
#         return jsonify({"error": "vLLM generation failed"}), 500


# @app.route("/stream/V2/", methods=['POST'])
# def stream_endpoint():
#     data = request.get_json(silent=True)
#     if not data:
#         return jsonify({"error": "Invalid or missing JSON"}), 400

#     query = data.get('query')
#     user_id = data.get('user_id')
#     index_type = data.get('index_type', 'knowledge_base_v1')
#     search_more = data.get('search_more', 0)
#     history = data.get('history', [])
#     file_names = data.get('file_names',[])
    
#     if not query or not user_id or not index_type:
#         return jsonify({"error": "Missing query, user_id or index_type"}), 400
#     # 第一层判断query和历史是否有联系，如果有联系则走模型，如果没有则走RAG
#     # 第二层如果提供了file_names则走知识库knowledge_base，如果没有则走长文档问答long_context
#     # 前端传参时，知识库的file_names是一个空列表，长文档问答的file_names是List[str]
#     if history:  
#         router_answer = service.llm_predictor.re_router(query)  
#         if router_answer == 'rag':  
#             return handle_search_and_streaming(user_id, query, index_type, search_more, history, file_names)  
#         elif router_answer == "translation":  
#             return service.llm_predictor.get_translation_answer(history)  
#         elif router_answer == "summary":  
#             return service.llm_predictor.get_summary_answer(history)  
#         elif router_answer == "table":  
#             return service.llm_predictor.get_table_answer(history)  
#     else:  
#         return handle_search_and_streaming(user_id, query, index_type, search_more, history, file_names)




# # 初始化 RougeReranker
# RougeReranker = RougeReranker(rouge_type='rouge-2', threshold=0.1)
     

# def handle_search_no_streaming(user_id, query, index_type, search_more, history, filenames=None):
#     try:
#         selected_rerank_chunks, _ = service.handle_search(user_id, query, index_type, search_more, filenames)
#         if selected_rerank_chunks is None:
#             logger.info("No related document information found")
#             return jsonify({"error": "No related information found, please upload new documents"}), 500
        
#         return service.handle_no_streaming(query, selected_rerank_chunks)
    
#     except Exception as e:
#         app.logger.error(f"vLLM generation failed: {str(e)}")
#         return jsonify({"error": "vLLM generation failed"}), 500


# @app.route("/json/V2/", methods=['POST'])
# def json_endpoint():
#     data = request.json
#     query = data.get('query')
#     user_id = data.get('user_id')
#     index_type = data.get('index_type', 'knowledge_base_v1')
#     search_more = data.get('search_more', 0)
#     history = data.get('history', []) 
#     file_names = data.get('file_names',[])

#     if not query or not user_id or not index_type:
#         return jsonify({"error": "Missing query, user_id or index_type"}), 400
    
#     if history:
#         router_answer = service.llm_predictor.re_router(query)        
#         if router_answer == 'rag':
#             response = handle_search_no_streaming(user_id, query, index_type, search_more, history, file_names)
#             json_file = service.handle_json(user_id, query, index_type, search_more, history, file_names)
#             json_data = json.loads(json_file.get_data(as_text=True))
#             return RougeReranker.rerank_jsondata(json_data, response)
#         else:
#             # 不走RAG返回一个空的json
#             return jsonify({}), 200
#     else:
#         response = handle_search_no_streaming(user_id, query, index_type, search_more, history, file_names)
#         json_file = service.handle_json(user_id, query, index_type, search_more, history, file_names)
#         json_data = json.loads(json_file.get_data(as_text=True))
#         return RougeReranker.rerank_jsondata(json_data, response)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=Config.STREAM_PORT)


import os
import re
import json
from loguru import logger
from config import Config
from flask_cors import CORS
from ESearch import Indexer
from rerank import Reranker, RougeReranker
from llm_vllm import LLMPredictor
from typing import List, Tuple, Dict
from flask import Flask, Response, request, jsonify, stream_with_context

os.environ["CUDA_VISIBLE_DEVICES"] = Config.RUN_DEVICES

class HybridSearchService:
    def __init__(self, es_hosts, embedding_model_path, model_name_or_path, reranker_model_path):
        self.indexer = Indexer(es_hosts, embedding_model_path)
        self.llm_predictor = LLMPredictor(model_name_or_path)
        self.reranker = Reranker(reranker_model_path)
        
    def es_search(self, user_id: str, query_str: str, index_type: str, filenames: List[str] = None):
        try:
            return self.indexer.search_with_filename(user_id, index_type, query_str, filenames), "success"
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [None] * 15, "es_search_failed"
        
    def handle_search(self, user_id: str, query: str, index_type: str, search_more: int = 0, filenames: List[str] = None):         
        try:
            query_group = self.llm_predictor.generate_queries(query)
        except Exception as e:
            logger.error(f"vLLM generation failed: {str(e)}")
            return None, None, "vllm_generation_failed"
        
        results = []
        for q in query_group:
            if q:
                result, status = self.es_search(user_id, q, index_type, filenames)
                if status == "es_search_failed":
                    return None, None, "es_search_failed"
                results.append(result)
        
        if not results:
            return None, None, "empty_database"

        recall_chunks = [chunk for result in results for chunk in result[0]]
        recall_chunks = list(set(recall_chunks))
        logger.info(f"针对问题:{query} 去重后的Recall chunks length: {len(recall_chunks)}")
        logger.info(f"针对问题:{query} 去重后的Recall chunks: {recall_chunks}")
        
        try:
            rerank_chunks = self.reranker.rerank(query, [recall_chunks], k=len(recall_chunks), threshold=0.5)
            logger.info(f"针对问题:{query} Rerank chunks length: {len(rerank_chunks)}")
            rerank_chunks = rerank_chunks[:15] # 最多返回15个chunk,不然要OOM
        except Exception as e:
            logger.error(f"Rerank model failed: {str(e)}")
            return None, None, "rerank_failed"
        
        if not rerank_chunks:
            logger.info(f"问题：{query} 在阈值为0.3时经过rerank后的chunks为空")
            return None, None, "empty_after_rerank"
                
        end_index = min((search_more + 1) * 5, len(rerank_chunks))
        selected_rerank_chunks = rerank_chunks[:end_index]
        logger.info(f"用户{user_id}的问题{query} 此次选择了Rerank中的第1到{end_index}的chunk进行RAG生成")
        logger.info(f"针对问题:{query} All Rerank results length: {len(rerank_chunks)}")
        logger.info(f"针对问题:{query} Selected Rerank results length: {len(selected_rerank_chunks)}")
        logger.info(f"针对问题:{query} Selected Rerank results: {selected_rerank_chunks}")     
        
        final_result = []
        seen_chunks = {}

        for chunk in selected_rerank_chunks:            
            for result in results:  
                chunks = result[0]
                if chunk in chunks:
                    index = chunks.index(chunk)
                    chunk_id = result[7][index]
                    seen_chunks[chunk_id] = {
                        "chunk_text": chunk,
                        "file_texts_path": result[1][index],
                        "file_name_md5_new": result[2][index],
                        "file_real_name": result[3][index],
                        "file_type": result[4][index],
                        "positions": result[5][index],
                        "page_no": result[6][index],
                        "chunk_id": chunk_id,
                        "mysql_id": result[8][index],
                        "file_from_folder": result[9][index],
                        "file_size": result[10][index],
                        "file_num_chars": result[11][index],
                        "file_upload_time": result[12][index],
                        "chunk_text_list": result[13][index],
                        "base64_list": result[14][index]
                    }
        final_result = list(seen_chunks.values())        
        final_data = json.dumps(final_result, ensure_ascii=False, indent=4)
        return selected_rerank_chunks, final_data, "success"
    
    def handle_no_streaming(self, query, rerank_chunks):
        try:
            response = self.llm_predictor.get_answer_from_pdf(rerank_chunks,query)
            return response
        except KeyboardInterrupt:
            return '[WARNING] Generation interrupted'
    
    def handle_streaming(self, query, rerank_chunks, history=None):
        @stream_with_context
        def generate():
            try:
                for new_text in self.llm_predictor._chat_stream(query, rerank_chunks):
                    yield new_text
            except KeyboardInterrupt:
                yield '[WARNING] Generation interrupted'    
        return Response(generate(), mimetype='text/event-stream')

    def handle_json(self, user_id, query, index_type, search_more=0, history=None, file_names=None):
        try:
            selected_rerank_chunks, final_data, status = self.handle_search(user_id, query, index_type, search_more, file_names)
            if status == "vllm_generation_failed":
                error_message = "vLLM is down. Please try again or contact support."
                logger.info("vLLM query generation failed")
                return jsonify({"error": error_message}), 500
            
            elif status == "es_search_failed":
                error_message = "Elasticsearch search failed. Please check the connection and try again."
                logger.info("Elasticsearch search failed")
                return jsonify({"error": error_message}), 500
            
            elif status == "empty_database":
                error_message = "No related information found in the database. Please upload new documents."
                logger.info("Database is empty or no matching documents found")
                return jsonify({"error": error_message}), 404
            
            elif status == "rerank_failed":
                error_message = "Reranking process failed. Please try again later."
                logger.info("Reranking process failed")
                return jsonify({"error": error_message}), 500
            
            elif status == "empty_after_rerank":
                error_message = "No results met the relevance threshold after reranking. Please try a different query."
                logger.info("No chunks passed the reranking threshold")
                return jsonify({"error": error_message}), 404
            
            elif status == "success":
                logger.info("Returning search results")
                return Response(final_data, mimetype='application/json')
            
            else:
                raise Exception("Unknown status returned from handle_search")

        except Exception as e:
            logger.error(f"Error in handle_json: {e}")
            return jsonify({"error": "An error occurred while processing the request"}), 500
    
app = Flask(__name__) 
CORS(app)
service = HybridSearchService(
    es_hosts=Config.ES_HOSTS,
    embedding_model_path=Config.EMBEDDING_MODEL_PATH,
    model_name_or_path=Config.MODEL_PATH,
    reranker_model_path=Config.RERANKER_MODEL_PATH
)

def handle_search_and_streaming(user_id, query, index_type, search_more, history, filenames=None):  
    try:  
        selected_rerank_chunks, final_data, status = service.handle_search(user_id, query, index_type, search_more, filenames) 
        if status != "success":
            error_messages = {
                "vllm_generation_failed": "vLLM is down. Please try again or contact support.",
                "es_search_failed": "Elasticsearch search failed. Please check the connection and try again.",
                "empty_database": "No related information found in the database. Please upload new documents.",
                "rerank_failed": "Reranking process failed. Please try again later.",
                "empty_after_rerank": "No results met the relevance threshold after reranking. Please try a different query."
            }
            return jsonify({"error": error_messages.get(status, "An unknown error occurred")}), 500
        
        # 如果有图片返回，不能再回答 "无答案。"
        answer = service.handle_no_streaming(query, selected_rerank_chunks)
        json_data = json.loads(final_data)
        
        base_64_list = []
        for i in json_data:
            if isinstance(i["base64_list"], list):
                base_64_list.extend(i["base64_list"])
            else:
                base_64_list.append(i["base64_list"])
                       
        base_64_list_setted = list(set(base_64_list))
        if len(base_64_list_setted) > 1 and answer == "无答案。":
            logger.info(f"图片数量：{len(base_64_list_setted)}")
            return "如图："   
        
            
        return service.handle_streaming(query, selected_rerank_chunks, history)
    except Exception as e:  
        app.logger.error(f"Error in handle_search_and_streaming: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request"}), 500
    
def handle_search_and_streaming_old(user_id, query, index_type, search_more, history, filenames=None):  
    try:  
        selected_rerank_chunks, final_data, status = service.handle_search(user_id, query, index_type, search_more, filenames) 
        if status != "success":
            error_messages = {
                "vllm_generation_failed": "vLLM is down. Please try again or contact support.",
                "es_search_failed": "Elasticsearch search failed. Please check the connection and try again.",
                "empty_database": "No related information found in the database. Please upload new documents.",
                "rerank_failed": "Reranking process failed. Please try again later.",
                "empty_after_rerank": "No results met the relevance threshold after reranking. Please try a different query."
            }
            return jsonify({"error": error_messages.get(status, "An unknown error occurred")}), 500
        
        return service.handle_streaming(query, selected_rerank_chunks, history)
    except Exception as e:  
        app.logger.error(f"Error in handle_search_and_streaming: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request"}), 500

@app.route("/stream/V3/", methods=['POST'])
def stream_endpoint():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    query = data.get('query')
    user_id = data.get('user_id')
    index_type = data.get('index_type', 'knowledge_base_v1')
    search_more = data.get('search_more', 0)
    history = data.get('history', [])
    file_names = data.get('file_names',[])
    
    if not query or not user_id or not index_type:
        return jsonify({"error": "Missing query, user_id or index_type"}), 400
    # 第一层判断query和历史是否有联系，如果有联系则走模型，如果没有则走RAG
    # 第二层如果提供了file_names则走知识库knowledge_base，如果没有则走长文档问答long_context
    # 前端传参时，知识库的file_names是一个空列表，长文档问答的file_names是List[str]    
    if history:  
        router_answer = service.llm_predictor.re_router(query)  
        if router_answer == 'rag':  
            return handle_search_and_streaming(user_id, query, index_type, search_more, history, file_names)  
        elif router_answer == "translation":  
            return service.llm_predictor.get_translation_answer(history)  
        elif router_answer == "summary":  
            return service.llm_predictor.get_summary_answer(history)  
        elif router_answer == "table":  
            return service.llm_predictor.get_table_answer(history)  
    else:  
        return handle_search_and_streaming(user_id, query, index_type, search_more, history, file_names)

# 初始化 RougeReranker
RougeReranker = RougeReranker(rouge_type='rouge-2', rouge_score_threshold=0.3)
     
def handle_search_no_streaming(user_id, query, index_type, search_more, history, filenames=None):
    try:
        selected_rerank_chunks, _, status = service.handle_search(user_id, query, index_type, search_more, filenames)
        if status != "success":
            error_messages = {
                "vllm_generation_failed": "vLLM is down. Please try again or contact support.",
                "es_search_failed": "Elasticsearch search failed. Please check the connection and try again.",
                "empty_database": "No related information found in the database. Please upload new documents.",
                "rerank_failed": "Reranking process failed. Please try again later.",
                "empty_after_rerank": "No results met the relevance threshold after reranking. Please try a different query."
            }
            return jsonify({"error": error_messages.get(status, "An unknown error occurred")}), 500
        
        return service.handle_no_streaming(query, selected_rerank_chunks)
    
    except Exception as e:
        app.logger.error(f"Error in handle_search_no_streaming: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request"}), 500

@app.route("/json/V3/", methods=['POST'])
def json_endpoint():
    data = request.json
    query = data.get('query')
    user_id = data.get('user_id')
    index_type = data.get('index_type', 'knowledge_base_v1')
    search_more = data.get('search_more', 0)
    history = data.get('history', []) 
    file_names = data.get('file_names',[])

    if not query or not user_id or not index_type:
        return jsonify({"error": "Missing query, user_id or index_type"}), 400
    
    if history:
        router_answer = service.llm_predictor.re_router(query)        
        if router_answer == 'rag':
            response = handle_search_no_streaming(user_id, query, index_type, search_more, history, file_names)
            json_file = service.handle_json(user_id, query, index_type, search_more, history, file_names)
            if isinstance(json_file, tuple):  # 如果返回的是错误信息
                return json_file
            json_data = json.loads(json_file.get_data(as_text=True))
            return service.reranker.rerank_jsondata_chunk(json_data, response)
        else:
            # 不走RAG返回一个空的json
            return jsonify({}), 200
    else:
        response = handle_search_no_streaming(user_id, query, index_type, search_more, history, file_names)
        json_file = service.handle_json(user_id, query, index_type, search_more, history, file_names)
        if isinstance(json_file, tuple):  # 如果返回的是错误信息
            return json_file
        json_data = json.loads(json_file.get_data(as_text=True))
        return service.reranker.rerank_jsondata_chunk(json_data, response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.STREAM_PORT, debug=True) 