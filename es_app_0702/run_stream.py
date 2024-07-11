import json
from loguru import logger
from config import Config
from flask_cors import CORS
from flask import Flask, request, jsonify
from search_service import HybridSearchService
from elasticsearch import ConnectionError, NotFoundError


# 初始化
def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)
    return app

app = create_app()

# 初始化服务
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


# 搜索和流处理函数
def handle_search_and_streaming(user_id, query, index_type, search_more, history, filenames=None):
    try:
        selected_rerank_chunks, final_data, status = service.handle_search(user_id, query, index_type, search_more, filenames)
        if status != "success":
            return handle_error(status) # 使用 handle_error 处理所有错误状态

        # 如果有图片返回，不能再回答 "无答案。"
        answer = service.handle_no_streaming(query, selected_rerank_chunks)
        json_data = json.loads(final_data)
        
        # 全组
        base_64_list = []
        for i in json_data:
            if isinstance(i["base64_list"], list):
                base_64_list.extend(i["base64_list"])
            else:
                base_64_list.append(i["base64_list"])
        
        # 只取第一组
        base_64_list = []
        if isinstance(json_data[0]["base64_list"], list):
            base_64_list.extend(json_data[0]["base64_list"])
        else:
            base_64_list.append(json_data[0]["base64_list"])

        base_64_list_setted = list(set(base_64_list))
        if len(base_64_list_setted) > 1 and answer == "无答案。":
            logger.info(f"图片数量：{len(base_64_list_setted)}")
            return "如图："

        return service.handle_streaming(query, selected_rerank_chunks, history)
    except Exception as e:
        app.logger.error(f"Error in handle_search_and_streaming: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request"}), 500

# 路由定义
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
    file_names = data.get('file_names', [])
    
    if not query or not user_id or not index_type:
        return jsonify({"error": "Missing query, user_id or index_type"}), 400

    # 第一层判断query和历史是否有联系，如果有联系则走模型，如果没有则走RAG
    # 第二层如果提供了file_names则走知识库knowledge_base，如果没有则走长文档问答long_context
    # 前端传参时，知识库的file_names是一个空列表，长文档问答的file_names是List[str]
    if history:
        router_answer = service.llm.re_router(query)
        if router_answer == 'rag':
            return handle_search_and_streaming(user_id, query, index_type, search_more, history, file_names)
        elif router_answer == "translation":
            return service.llm.get_translation_answer(history)
        elif router_answer == "summary":
            return service.llm.get_summary_answer(history)
        elif router_answer == "table":
            return service.llm.get_table_answer(history)
    else:
        return handle_search_and_streaming(user_id, query, index_type, search_more, history, file_names)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.STREAM_PORT)