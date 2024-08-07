from loguru import logger
from bm25 import BM25Ranker_attach
from agent_rerank import AgentReranker

class Search_Agent:
    
    def __init__(self, indexer):
        self.indexer = indexer
        self.bm25_ranker = BM25Ranker_attach()
        self.agent_reranker = AgentReranker()
        
        
    def sort_chunks(self, chunks):
        def sort_key(chunk):
            id_str = chunk['_source']['file_docai_json']['id']
            return int(id_str.split('-')[0])
        
        sorted_chunks = sorted(chunks, key=sort_key)
        return sorted_chunks
        
    # aux_func
    def re_rank_by_weighted_scores(self, sorted_chunks):
        # 提取re_score和bm25_score
        re_scores = [chunk['_source']['file_docai_json']['re_score'] for chunk in sorted_chunks]
        bm25_scores = [chunk['_source']['file_docai_json']['bm25_score'] for chunk in sorted_chunks]

        # 归一化re_scores
        re_min, re_max = min(re_scores), max(re_scores)
        re_scores_normalized = [(score - re_min) / (re_max - re_min) if re_max > re_min else 0 for score in re_scores]

        # 归一化bm25_scores
        bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        bm25_scores_normalized = [(score - bm25_min) / (bm25_max - bm25_min) if bm25_max > bm25_min else 0 for score in bm25_scores]

        # 计算合并得分
        weighted_scores = [0.9 * re + 0.0000001 * bm for re, bm in zip(re_scores_normalized, bm25_scores_normalized)]

        # 为每个chunk添加合并得分，并重新排序
        for chunk, score in zip(sorted_chunks, weighted_scores):
            chunk['_source']['file_docai_json']['combined_score'] = score

        # 根据合并得分降序排序
        sorted_chunks.sort(key=lambda x: x['_source']['file_docai_json']['combined_score'], reverse=True)

        return sorted_chunks
        
    def return_several_file_position_information(self, user_id, index_type, query, statement, file_names=None):
        
        chunk_text = []
        file_texts_path = []  
        file_name_md5_new = []
        file_real_name = []
        file_type = []
        positions = []  
        page_no = []
        chunk_id = []
        mysql_id = []
        file_from_folder = []
        file_size = []
        file_num_chars = []
        file_upload_time = []
        text_list = []
        base64_list = []

        chunks = self.indexer.search_chunk_by_query_attach(user_id, index_type, statement, file_names)
        sorted_docs_temp_1 = self.bm25_ranker.rank(statement, chunks)
        sorted_docs_temp_2 = self.agent_reranker.rerank(statement, sorted_docs_temp_1)
        chunks = self.re_rank_by_weighted_scores(sorted_docs_temp_2)
        try:
            chunks = chunks[0:6]
        except:
            chunks = chunks[:]
        for chunk in chunks:
            logger.info(f"相关的chunk内容为：{chunk['_source']['file_docai_json']['text']}")
        for i, chunk in enumerate(chunks):
            
            chunk_text.append(chunk['_source']['file_docai_json']['text'])
            file_texts_path.append(chunk['_source']['file_texts_path']) 
            file_name_md5_new.append(chunk['_source']['file_name_md5_new'])
            file_real_name.append(chunk['_source']['file_real_name'])
            file_type.append(chunk['_source']['file_type'])
            positions.append(chunk['_source']['file_docai_json']['positions']) 
            page_no.append(chunk['_source']['file_docai_json']['page_no'])
            chunk_id.append(chunk['_source']['file_docai_json']['id'])
            mysql_id.append(chunk['_source']['mysql_id'])
            file_from_folder.append(chunk['_source']['file_from_folder'])
            file_size.append(chunk['_source']['file_size'])
            file_num_chars.append(chunk['_source']['file_num_chars'])
            file_upload_time.append(chunk['_source']['file_upload_time'])
            text_list.append(chunk['_source']['file_docai_json']['text_list'])
            base64_list.append(chunk['_source']['file_docai_json']['base64_list'])
        
        chunk_list = [chunk['_source'] for chunk in chunks]
        page_no_list = [chunk['file_docai_json']['page_no'] for chunk in chunk_list]
        attach_list = [chunk['file_docai_json']['attach_text'].split('的部分内容为')[0] for chunk in chunk_list]
        relevent_text_list = [chunk['file_docai_json']['text'] for chunk in chunk_list]
        describe = ''
        
        for page in page_no_list:
            logger.info(f"获取的page信息为：{page}")
        
        for i in range(len(chunk_list)):
            describe += f"第{i+1}个相关信息是{relevent_text_list[i]}，具体位置是{attach_list[i]}，处于整个文件的第{page_no_list[i]}页。\n"
        logger.info(f"融合的描述为:{describe}")
        return ["搜索",f"问题：\n{query}\n根据下面几个描述信息：\n{describe}请从描述信息中判断并选择有用的内容。例如，如果问题只询问是哪个文件你只用回答文件名即可，不需要回答等无关信息，并在文件名重复时进行去重。你的回答是一个简单的陈述，不要包含额外的分析，务必注意，如果所给材料无法回答用户问题，只输出无答案。你的输出为：",chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, text_list, base64_list]
    
    
    
    
    
    
    
    # def return_one_file_position_information(self, user_id, index_type, query, statement,file_names=None):
        
    #     chunk_text = []
    #     file_texts_path = [] 
    #     file_name_md5_new = []
    #     file_real_name = []
    #     file_type = []
    #     positions = []  
    #     page_no = []
    #     chunk_id = []
    #     mysql_id = []
    #     file_from_folder = []
    #     file_size = []
    #     file_num_chars = []
    #     file_upload_time = []
    #     text_list = []
    #     base64_list = []
        
    #     chunks= self.indexer.search_chunk_by_query_attach(user_id, index_type, statement ,file_names)
        
        
    #     # for i, chunk in enumerate(chunks):
    #     chunk = chunks[0]
    #     chunk_text.append(chunk['_source']['file_docai_json']['text'])
    #     file_texts_path.append(chunk['_source']['file_texts_path'])   # 这个路径是文本文件的路径
    #     file_name_md5_new.append(chunk['_source']['file_name_md5_new'])
    #     file_real_name.append(chunk['_source']['file_real_name'])
    #     file_type.append(chunk['_source']['file_type'])
    #     positions.append(chunk['_source']['file_docai_json']['positions']) 
    #     page_no.append(chunk['_source']['file_docai_json']['page_no'])
    #     chunk_id.append(chunk['_source']['file_docai_json']['id'])
    #     mysql_id.append(chunk['_source']['mysql_id'])
    #     file_from_folder.append(chunk['_source']['file_from_folder'])
    #     file_size.append(chunk['_source']['file_size'])
    #     file_num_chars.append(chunk['_source']['file_num_chars'])
    #     file_upload_time.append(chunk['_source']['file_upload_time'])
    #     text_list.append(chunk['_source']['file_docai_json']['text_list'])
    #     base64_list.append(chunk['_source']['file_docai_json']['base64_list'])
        
    #     chunk = chunks[0]['_source']
    #     page_num = chunk['file_docai_json']['page_no']
    #     attach_text = chunk['file_docai_json']['attach_text'].split('的部分内容为')[0]
    #     text = chunk['file_docai_json']['text']
    #     describe = f"相关信息是{text}，具体位置是{attach_text}，处于整个文件的第{page_num}页。\n"
    #     # return ["搜索",f"问题：\n{query}\n全部可能用到的描述信息为：\n{describe}根据上述问题，请从描述信息中判断并选择有用的内容。例如，如果问题只询问是哪个文件你只用回答文件名即可，不需要回答章节和页码等无关信息。你的回答是一个简单的陈述，不要包含额外的分析，务必注意，如果所给材料无法回答用户问题，只输出无答案。你的输出为：",chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, text_list, base64_list]
    #     return ["搜索",f"问题：\n{query}\n全部可能用到的描述信息为：\n{describe}根据上述问题，请从描述信息中判断并选择有用的内容。例如，如果问题只询问是哪个文件你只用回答文件名即可，不需要回答无关信息。你的回答是一个简单的陈述，不要包含额外的分析，务必注意，如果所给材料无法回答用户问题，只输出无答案。你的输出为：",chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, text_list, base64_list]
