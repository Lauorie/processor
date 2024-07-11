import json
from loguru import logger
from config import Config
from flask_cors import CORS
from flask import Flask, request, jsonify
from search_service import HybridSearchService

# 初始化
def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)
    return app

app = create_app() 


# initialize the service
service = HybridSearchService(reranker_threshold=0.3)

# 错误处理函数
def handle_error(status: str):
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

def handle_search_no_streaming(user_id, query, index_type, search_more, history, filenames=None):
    try:
        selected_rerank_chunks, _, status = service.handle_search(user_id, query, index_type, search_more, filenames)
        if status != "success":
            return handle_error(status)        
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
        router_answer = service.llm.re_router(query)        
        if router_answer == 'rag':
            response = handle_search_no_streaming(user_id, query, index_type, search_more, history, file_names)
            json_file = service.handle_json(user_id, query, index_type, search_more, history, file_names)
            if isinstance(json_file, tuple):  # 如果返回的是错误信息
                return json_file
            json_data = json.loads(json_file.get_data(as_text=True))
            return service.reranker.rerank_combined_chunk(json_data, response)
        else:
            # 不走RAG返回一个空的json
            return jsonify({}), 200
    else:
        response = handle_search_no_streaming(user_id, query, index_type, search_more, history, file_names)
        json_file = service.handle_json(user_id, query, index_type, search_more, history, file_names)
        if isinstance(json_file, tuple):  # 如果返回的是错误信息
            return json_file
        json_data = json.loads(json_file.get_data(as_text=True))
        return service.reranker.rerank_combined_chunk(json_data, response)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.JSON_PORT, debug=True)