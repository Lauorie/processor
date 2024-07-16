import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from flask_cors import CORS
from Router import Router_agent
from Router_attribute import Router_agent_attribute
from config import Config
from flask import Flask, Response, request, jsonify, stream_with_context


class Long_Context_Processer:
    def __init__(self, config):
        self.config = config 
        self.router = Router_agent(config=self.config)
        self.router_attribute = Router_agent_attribute(config=self.config)
    
    
    def agent_output(self, user_id, index_type, query, history_list=None, search_more=None, file_names=None):
        @stream_with_context
        def stream_output():
            return self.router.long_context_process(query=query, user_id=user_id,index_type=index_type, history_list=history_list, search_more=search_more, file_names=file_names)
        return Response(stream_output(),mimetype='text/event-stream')
    
    def return_attribute(self, user_id, index_type, query, history_list=None, search_more=None, file_names=None):
        response = self.router_attribute.long_context_process(query=query, user_id=user_id,index_type=index_type, history_list=history_list, search_more=search_more, file_names=file_names)
        if response[0] == "搜索":
            try:
                attribute_list = [{
                "chunk_text": response[-15],
                "file_texts_path":response[-14],
                "file_name_md5_new":response[-13],
                "file_real_name":response[-12],
                "file_type":response[-11],
                "positions":response[-10],
                "page_no":response[-9],
                "chunk_id":response[-8],
                "mysql_id":response[-7],
                "file_from_folder":response[-6],
                "file_size":response[-5],
                "file_num_chars":response[-4],
                "file_upload_time":response[-3],
                "chunk_text_list":response[-2],
                "base64_list":response[-1],
            }]
            except:
                attribute_list = []
        else:
            try:
                attribute_list = [{
                "chunk_text": [response[-15][0]],
                "file_texts_path":[response[-14][0]],
                "file_name_md5_new":[response[-13][0]],
                "file_real_name":[response[-12][0]],
                "file_type":[response[-11][0]],
                "positions":[response[-10][0]],
                "page_no":[response[-9][0]],
                "chunk_id":[response[-8][0]],
                "mysql_id":[response[-7][0]],
                "file_from_folder":[response[-6][0]],
                "file_size":[response[-5][0]],
                "file_num_chars":[[response[-4][0]]],
                "file_upload_time":[response[-3][0]],
                "chunk_text_list":[response[-2][0]],
                "base64_list":[response[-1][0]],
            }]
            except:
                attribute_list = []
        return json.dumps(attribute_list, indent=4,ensure_ascii=False)
    
app = Flask(__name__)
CORS(app)

processor = Long_Context_Processer(Config)

@app.route('/agent/V1/', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query')
    user_id = data.get('user_id')
    index_type = data.get('index_type', 'long_context_v1')
    search_more = data.get('search_more', 0)
    history = data.get('history', []) 
    file_names = data.get('file_names',[])

    if not query or not user_id or not index_type:
        return jsonify({"error": "Missing required fields"}), 400

    output = processor.agent_output(index_type=index_type, user_id=user_id, query=query, history_list=history, search_more=search_more, file_names=file_names)
    # output = processor.return_attribute(index_type=index_type, user_id=user_id, query=query, history_list=history, search_more=search_more, file_names=file_names)
    return output


@app.route('/agent_attribute/V1/', methods=['POST'])
def process_attribute():
    data = request.json
    query = data.get('query')
    user_id = data.get('user_id')
    index_type = data.get('index_type', 'long_context_v1')
    search_more = data.get('search_more', 0)
    history = data.get('history', []) 
    file_names = data.get('file_names',[])

    if not query or not user_id or not index_type:
        return jsonify({"error": "Missing required fields"}), 400

    # output = processor.agent_output(index_type=index_type, user_id=user_id, query=query, history_list=history, search_more=search_more, file_names=file_names)
    output = processor.return_attribute(index_type=index_type, user_id=user_id, query=query, history_list=history, search_more=search_more, file_names=file_names)
    return output


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.AGENT_PORT, debug=True)
    
