import json
from loguru import logger
from config import Config
from flask_cors import CORS
from flask import Flask, request, jsonify
from search_service import HybridSearchService


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)
    return app

app = create_app()

service = HybridSearchService(reranker_threshold=0.4)

@app.route("/ai_search/V2/", methods=['POST'])
def json_endpoint():
    try:
        data = request.json
        query = data.get('query')
        user_id = data.get('user_id')
        index_type = data.get('index_type', 'knowledge_base_v1')
    
        if not query or not user_id or not index_type:
            return jsonify({"error": "Missing query, user_id or index_type"}), 400
    
        return service.handle_json(user_id, query, index_type)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.AI_SEARCH_PORT)