import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
from loguru import logger
from bm25 import BM25Ranker_attach
from transformers import AutoTokenizer
from agent_rerank import AgentReranker
from config import Config

class Backup_Agetnt:
    def __init__(self, indexer):
        self.indexer = indexer
        self.bm25_ranker = BM25Ranker_attach()
        self.agent_reranker = AgentReranker()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME_OR_PATH, trust_remote_code=True)
    
    # aux_func
    def adjust_length(self, text):
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 7000:
            return text
        else:
            tokens_a = tokens[:7000]
            str_a = self.tokenizer.decode(tokens_a, skip_special_tokens=True)
            return str_a
    
    # aux_func   
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
        weighted_scores = [0.3 * re + 0.8 * bm for re, bm in zip(re_scores_normalized, bm25_scores_normalized)]

        # 为每个chunk添加合并得分，并重新排序
        for chunk, score in zip(sorted_chunks, weighted_scores):
            chunk['_source']['file_docai_json']['combined_score'] = score

        # 根据合并得分降序排序
        sorted_chunks.sort(key=lambda x: x['_source']['file_docai_json']['combined_score'], reverse=True)

        return sorted_chunks
    
    def process(self, user_id, index_type, query, file_names=None):
        chunks = self.indexer.search_chunk_by_query_attach(user_id, index_type, query, file_names)
        chunks = self.agent_reranker.rerank(query, chunks)
        id = chunks[0]['_source']['file_docai_json']['id']
        file_name_md5 = chunks[0]['_source']['file_name_md5_new']
        
        full_chunks = self.indexer.serach_chunk_by_file_name(user_id, index_type, file_name_md5)
        full_chunks = self.sort_chunks(full_chunks)
        
        for s, chunk in enumerate(full_chunks):
            if chunk['_source']['file_docai_json']['id'] == id:
                start = s
                break
        try:
            for e, chunk in enumerate(full_chunks[start+1:]):
                if chunk['_source']['file_docai_json']['type'] == 'TITLE':
                    end = e+s
                    break
        except:
            for e, chunk in enumerate(full_chunks[start:]):
                if chunk['_source']['file_docai_json']['type'] == 'TITLE':
                    end = e+s
                    break
        try:
            chunks = full_chunks[start:end+1]
        except:
            chunks = full_chunks[start:]
            
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
        
        for i, chunk in enumerate(chunks):
            if chunk['_source']['file_docai_json']['type'] in ['SECTION-TEXT', 'TITLE', 'CAPTION','SECTION-TITLE']:
                try:
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
                except:
                    pass
        full_text_of_title = ''
        for text in chunk_text:
            full_text_of_title += text
        full_text_of_title = self.adjust_length(full_text_of_title)
        return ['备用', full_text_of_title, chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, text_list, base64_list]

    