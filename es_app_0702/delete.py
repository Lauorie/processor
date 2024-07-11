from config import Config
from loguru import logger
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch, NotFoundError, ConnectionError

logger.add(Config.DELETE_LOG_PATH)
app = Flask(__name__)
es = Elasticsearch(hosts=Config.ES_HOSTS)

@app.route('/delete_files/V1/', methods=['POST'])
def delete_files():
    try:
        # 解析请求数据
        data = request.get_json()
        user_id = data.get('user_id')
        file_names_md5_new = data.get('file_names_md5_new', [])
        file_used_for = data.get('file_used_for')  # 知识库还是长文档
        
        if not file_names_md5_new:
            return jsonify({"status": "failure", "message": "No files specified for deletion."}), 400
        
        index_name = f"{file_used_for}_{user_id}"
        deleted_files = []
        not_found_files = []

        for file_name_md5_new in file_names_md5_new:
            # 根据 user_id 和 file_name_md5_new 构建查询
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"user_id": user_id}},
                            {"match": {"file_name_md5_new": file_name_md5_new}},
                            {"match": {"file_used_for": file_used_for}}
                        ]
                    }
                }
            }
            
            # 检查是否找到了匹配的文档
            response = es.search(index=index_name, body=query)
            if response['hits']['total']['value'] > 0:
                # 如果文档存在，则执行删除操作
                delete_response = es.delete_by_query(index=index_name, body=query)
                deleted_files.append(file_name_md5_new)
                logger.info(f"{user_id} deleted file {file_name_md5_new} from {file_used_for}.")
            else:
                # 如果文档不存在，记录未找到的文件
                not_found_files.append(file_name_md5_new)
                logger.warning(f"{user_id} tried to delete file {file_name_md5_new} from {file_used_for}, but file not found.")
        
        if not_found_files:
            return jsonify({"status": "partial success", "message": "Some files were not found.", "deleted_files": deleted_files, "not_found_files": not_found_files}), 207
        else:
            return jsonify({"status": "success", "message": "All files deleted.", "deleted_files": deleted_files}), 200
    
    except NotFoundError:
        # 处理索引未找到的情况
        return jsonify({"status": "failure", "message": f"Index '{index_name}' not found."}), 404
    
    except ConnectionError:
        # 处理连接错误
        return jsonify({"status": "failure", "message": "Elasticsearch connection error."}), 500
    
    except Exception as e:
        # 处理其他异常
        return jsonify({"status": "failure", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.DELETE_PORT, debug=True)