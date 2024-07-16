import os
import re
import json
from loguru import logger
from config import Config
from flask_cors import CORS
from esearch import Indexer
from functools import lru_cache
from rerank import Reranker, RougeReranker
from vllm_llm import LLM
from typing import List, Tuple, Dict, Optional
from flask import Flask, Response, request, jsonify, stream_with_context
from elasticsearch import ConnectionError, NotFoundError


class HybridSearchService:
    def __init__(self, reranker_threshold: float = 0.7):
        self.indexer = Indexer()
        self.llm = LLM()
        self.reranker = Reranker()
        self.reranker_threshold = reranker_threshold
        
    def es_search(self, user_id: str, query_str: str, index_type: str, filenames: List[str] = None):
        try:
            return self.indexer.search_with_filename(user_id, index_type, query_str, filenames), "success"
        except ConnectionError as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            return [None] * 15, "es_search_failed"
        except NotFoundError as e:
            logger.error(f"Elasticsearch index not found: {e}")
            return [None] * 15, "es_index_not_found"
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [None] * 15, "es_search_failed"
        
        
    def handle_search(self, user_id: str, query: str, index_type: str, search_more: int = 0, filenames: List[str] = None):         
        # query重写
        try:
            query_group = self.llm.generate_queries(query)
        except Exception as e:
            logger.error(f"vLLM generation failed: {str(e)}")
            return None, None, "vllm_generation_failed"
        
        results = []
        for q in query_group:
            if q:
                result, status = self.es_search(user_id, q, index_type, filenames)
                if status != "success": 
                    return None, None, status 
                results.append(result)
        
        if not results:
            return None, None, "empty_database"

        recall_chunks = [chunk for result in results for chunk in result[0]]
        recall_chunks = list(set(recall_chunks))
        logger.info(f"针对问题:{query} 去重后的Recall chunks length: {len(recall_chunks)}")
        logger.info(f"针对问题:{query} 去重后的Recall chunks: {recall_chunks}")
        
        try:
            rerank_chunks = self.reranker.rerank(query, [recall_chunks], k=len(recall_chunks), threshold=self.reranker_threshold)
            logger.info(f"针对问题:{query} Rerank chunks length: {len(rerank_chunks)}")
            rerank_chunks = rerank_chunks[:15] # 最多返回15个chunk,不然要OOM
        except Exception as e:
            logger.error(f"Rerank model failed: {str(e)}")
            return None, None, "rerank_failed"
        
        if not rerank_chunks:
            logger.info(f"问题：{query} 在阈值为{self.reranker_threshold}时经过rerank后的chunks为空")
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
            response = self.llm.get_answer_from_pdf(rerank_chunks,query)
            return response
        except KeyboardInterrupt:
            return '[WARNING] Generation interrupted'
    
    def handle_streaming(self, query, rerank_chunks, history=None):
        @stream_with_context
        def generate():
            try:
                for new_text in self.llm._chat_stream(query, rerank_chunks):
                    yield new_text
            except KeyboardInterrupt:
                yield '[WARNING] Generation interrupted'    
        return Response(generate(), mimetype='text/event-stream')
    
    def handle_json(
            self,
            user_id: str,
            query: str,
            index_type: str,
            search_more: int = 0,
            history=None,
            file_names: Optional[List[str]] = None
        ) -> Response:
        try:
            selected_rerank_chunks, final_data, status = self.handle_search(user_id, query, index_type, search_more, file_names)

            if status != "success":
                return self._handle_error(status)

            logger.info("Returning search results")
            return Response(final_data, mimetype='application/json')

        except Exception as e:
            logger.error(f"Error in handle_json: {e}")
            return jsonify({"error": "An error occurred while processing the request"}), 500

    def _handle_error(self, status: str) -> Response:
        error_messages = {
            "vllm_generation_failed": "vLLM is down. Please try again or contact support.",
            "es_search_failed": "Elasticsearch search failed. Please check the connection and try again.",
            "es_index_not_found": "Elasticsearch index not found. Please upload new documents and try again.",
            "rerank_failed": "Reranking process failed. Please try again later.",
            "empty_database": "No related information found in the database. Please upload new documents and try again.",
            "empty_after_rerank": "No results met the relevance threshold after reranking. Please try a different query."
        }

        error_message = error_messages.get(status, "An unknown error occurred")
        logger.info(error_message)

        status_code = 404 if status in ["empty_database", "empty_after_rerank", "es_index_not_found"] else 500
        return jsonify({"error": error_message}), status_code