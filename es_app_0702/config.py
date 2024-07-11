class Config:
    ES_HOSTS = ["http://localhost:9250"] # ["http://10.176.50.48:5079"]  ["http://localhost:9250"] 5079是47的es端口，9250是230的es端口
    EMBEDDING_MODEL_PATH = "/root/web_demo/HybirdSearch/models/models--iampanda--zpoint_large_embedding_zh" #"/root/web_demo/HybirdSearch/models/models--infgrad--stella-large-zh-v3-1792d"
    MODEL_NAME_OR_PATH = '/root/web_demo/HybirdSearch/models/models--Qwen--Qwen1.5-14B-Chat'  
    RERANKER_MODEL_PATH = '/root/web_demo/HybirdSearch/models/models--BAAI--bge-reranker-large'
    
    
    # logs config
    RUN_LOG_PATH = "/root/web_demo/HybirdSearch/es_app_0517/logs/run.log"
    UPLOAD_LOG_PATH = "/root/web_demo/HybirdSearch/es_app_0517/logs/upload.log"
    DELETE_LOG_PATH = "/root/web_demo/HybirdSearch/es_app_0517/logs/delete.log"
    MODEL_LOG_PATH = "/root/web_demo/HybirdSearch/es_app_0517/logs/model.log"
    
    # port config
    STREAM_PORT = 5066
    JSON_PORT = 5067
    AGENT_PORT = 5058
    UPLOAD_PORT = 5052
    DELETE_PORT = 5053
    AI_SEARCH_PORT = 5054
    AI_WRITING_PORT = 5056
    
    
    # vllm config
    API_KEY = "123"
    BASE_URL = "http://10.176.50.47:34066/v1"#"http://10.176.50.17:5026/v1"
    MODEL_PATH = '/root/app/models/models--Qwen--Qwen1.5-14B-Chat' 
    
    # 保存生成的数据
    RAG_DATA_PATH = '/root/web_demo/HybirdSearch/es_app_0517/RAG_query_answer.json'
    
    # GPU卡分配
    # AI_SEARCH_DEVICES = '0,1'
    # RUN_DEVICES = '2,3,4,5'
    # UPLOAD_DEVICES = '6,7'
    
    # embedding reranker config
    EMBEDDING_SERVER_PATH = 'models/models--iampanda--zpoint_large_embedding_zh'
    EMBEDDING_SERVER_URL = 'http://10.176.50.17:5023/embeddings'
    
    RERANKER_SERVER_PATH = 'models/models--BAAI--bge-reranker-large'
    RERANKER_SERVER_URL = 'http://10.176.50.17:5023/rerank'