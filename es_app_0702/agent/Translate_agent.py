import re
from loguru import logger
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bm25 import BM25Ranker_attach
from transformers import AutoTokenizer
from agent_rerank import AgentReranker
from config import Config

class Translate_Agetnt:
    def __init__(self, indexer):
        self.indexer = indexer
        self.bm25_ranker = BM25Ranker_attach()
        self.agent_reranker = AgentReranker()
        
    # aux_func   
    def sort_chunks(self, chunks):
        def sort_key(chunk):
            id_str = chunk['_source']['file_docai_json']['id']
            return int(id_str.split('-')[0])
        sorted_chunks = sorted(chunks, key=sort_key)
        return sorted_chunks
    
    # aux_func
    def remove_duplicates(self,chunks):
        seen_ids = set()
        unique_chunks = []
        
        for chunk in chunks:
            chunk_id = chunk['_source']['file_docai_json']['id']
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    # aux_func
    def remove_duplicate_id(self,chunks):
        processing_id = None
        for chunk in chunks:
            processing_id = chunk['_source']['file_docai_json']['id']
            
    # aux_func
    def re_rank_by_weighted_scores(self, sorted_chunks):
        re_scores = [chunk['_source']['file_docai_json']['re_score'] for chunk in sorted_chunks]
        bm25_scores = [chunk['_source']['file_docai_json']['bm25_score'] for chunk in sorted_chunks]

        re_min, re_max = min(re_scores), max(re_scores)
        re_scores_normalized = [(score - re_min) / (re_max - re_min) if re_max > re_min else 0 for score in re_scores]

        bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        bm25_scores_normalized = [(score - bm25_min) / (bm25_max - bm25_min) if bm25_max > bm25_min else 0 for score in bm25_scores]

        weighted_scores = [0.9 * re + 0.0000000001 * bm for re, bm in zip(re_scores_normalized, bm25_scores_normalized)]

        for chunk, score in zip(sorted_chunks, weighted_scores):
            chunk['_source']['file_docai_json']['combined_score'] = score

        sorted_chunks.sort(key=lambda x: x['_source']['file_docai_json']['combined_score'], reverse=True)

        return sorted_chunks
    
    #aux_func
    def concatenate_strings(self, str_list):
        result = []
        current_str = ""
        
        for s in str_list:
            if len(current_str) < 200:
                if len(current_str) + len(s) < 200:
                    current_str += s 
                else:
                    result.append(current_str)
                    current_str = s
            else:
                result.append(current_str)
                current_str = s
        
        if current_str:
            result.append(current_str)
        
        processed_result = []
        for item in result:
            if item.endswith('\n'):
                processed_item = item[:-1].replace('\n', '') + '\n'
            else:
                processed_item = item.replace('\n', '')
            
            processed_result.append(processed_item)
        
        return processed_result
    
    # 翻译X部分的这一页/翻译X文件的第X页
    def return_page(self, user_id, index_type, query, statement, page_num=None,file_names=None):
        
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
        logger.info(f"完成重排")
        for chunk in chunks:
            logger.info(f"-----用于召回的chunk的attach_text为：{chunk['_source']['file_docai_json']['attach_text']}")
        file_name_md5 = chunks[0]['_source']['file_name_md5_new']
        chunk = chunks[0]['_source']['file_docai_json']
        # logger.info(f"-----用于召回的chunk的text为：{chunk['text']}")
        
        if page_num == None:
            page_num = chunk['page_no']
            logger.info(f"-----用于召回的chunk的页码为：{chunk['page_no']}")
            chunks = self.indexer.serach_chunk_by_file_name_and_page(user_id, index_type, file_name_md5, page_num)
        else: 
            chunks = self.indexer.serach_chunk_by_file_name_and_page(user_id, index_type, file_name_md5, page_num)
            
        chunks = self.sort_chunks(chunks)
        for i, chunk in enumerate(chunks):
            if chunk['_source']['file_docai_json']['type'] in ['SECTION-TEXT','TITLE','CAPTION','SECTION-TITLE']:
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
            elif chunk['_source']['file_docai_json']['type'] == 'IMAGE':
                pass
            
        full_text = ''
        for text in chunk_text:
            full_text += text
            
        chunk_text = self.concatenate_strings(chunk_text)
        return ['翻译页', full_text, chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, text_list, base64_list]
            
    
    # 翻译X章节
    def return_title_content(self, user_id, index_type, query, statement, file_names=None):
        
        # 获取文件名file_name_md5
        chunks = self.indexer.search_chunk_by_query_attach(user_id, index_type, statement, file_names)
        file_name_md5 = chunks[0]['_source']['file_name_md5_new']
        chunks_backup = self.agent_reranker.rerank(statement, chunks)
        id_backup = chunks_backup[0]['_source']['file_docai_json']['id']
        text_backup = chunks_backup[0]['_source']['file_docai_json']['text']

        # 获取该file_name_md5下所有title和section-title类型的chunk
        try:
            chunks_title = self.indexer.search_chunk_by_title_filename(user_id, index_type, file_name_md5)
        except:
            logger.info(f"根据title检索失败")
            
        try:
            chunks_section_title = self.indexer.search_chunk_by_section_title_filename(user_id, index_type, file_name_md5)
        except:
            logger.info(f"根据section_title检索失败")

        chunks = chunks_title + chunks_section_title
            
        # 检索最相关title并得到其id
        logger.info(f"statement为:{statement}")
        sorted_docs_temp_1 = self.bm25_ranker.rank(statement, chunks)
        sorted_docs_temp_2 = self.agent_reranker.rerank(statement, sorted_docs_temp_1)
        sorted_docs = self.re_rank_by_weighted_scores(sorted_docs_temp_2)
        for index, chunk in enumerate(sorted_docs):
            logger.info(f"reranker排序的章节名为{chunk['_source']['file_docai_json']['text']},其类型为{chunk['_source']['file_docai_json']['type']}")
            if index > 9:
                break
        if sorted_docs[0]['_source']['file_docai_json']['id'].split('-')[0] == '0':
            try:
                id = sorted_docs[1]['_source']['file_docai_json']['id']
                type = sorted_docs[1]['_source']['file_docai_json']['type']
            except:
                id = sorted_docs[0]['_source']['file_docai_json']['id']
                type = sorted_docs[0]['_source']['file_docai_json']['type']
        else:
            id = sorted_docs[0]['_source']['file_docai_json']['id']
            type = sorted_docs[0]['_source']['file_docai_json']['type']
        file_name_md5 = sorted_docs[0]['_source']['file_name_md5_new']
        logger.info(f"type为{type}")
        logger.info(f"id为{id}")
        
        # 根据id和文件名重新构造这一章节
        if self.bm25_ranker.query_text_check(query=query, text=sorted_docs[0]['_source']['file_docai_json']['text']):
            logger.info(f"query和text的匹配度不足采取备用方案")
            logger.info(f"取到第一相关的文本为:\n{text_backup}")
            full_chunks = self.indexer.serach_chunk_by_file_name(user_id, index_type, file_name_md5)
            full_chunks = self.sort_chunks(full_chunks)
            
            
            for s, chunk in enumerate(full_chunks):
                if chunk['_source']['file_docai_json']['id'] == id_backup:
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
        
        if type == 'SECTION-TITLE':
            chunks_section = self.indexer.serach_chunk_by_file_name_and_parentid(user_id, index_type, file_name_md5, parent_id=id)
            chunks = self.sort_chunks(chunks_section)
            if len(chunks)<=1:
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
            
        if type == 'TITLE':
            start = None
            end = None
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
        
        logger.info(f"-----------------{len(chunks)}")
            
        
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
        
        chunk_text = self.concatenate_strings(chunk_text)
        return ['翻译章', full_text_of_title, chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, text_list, base64_list]
    
    
    # 直接翻译chunk的内容
    def return_chunk(self, user_id, index_type, query, statement, file_names=None):
        chunks = self.indexer.search_chunk_by_query_text(user_id, index_type, statement, file_names)
        chunks = self.agent_reranker.rerank(statement, chunks)
        chunk = chunks[0]
        text = chunk['_source']['file_docai_json']['text']
        
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
        
        chunk_text = self.concatenate_strings(chunk_text)
        return ['翻译chunk',text,chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, text_list, base64_list]
    
    # 翻译全文
    def return_full(self, user_id, index_type, query, statement,file_names=None):
        chunks = self.indexer.search_chunk_by_query_text(user_id, index_type, statement,file_names)
        chunk = chunks[0]['_source']['file_docai_json']
        file_name_md5 = chunks[0]['_source']['file_name_md5_new']
        
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
        
        chunks = self.indexer.serach_chunk_by_file_name(user_id, index_type, file_name_md5)
        chunks = self.sort_chunks(chunks)
        chunks = self.remove_duplicates(chunks)
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
            else:
                pass
        full_text = ''
        for text in chunk_text:
            full_text += text
        for i,text in enumerate(chunk_text):
            logger.info(f"文本：{text}")
            logger.info(f"id为：{chunk_id[i]}")
            logger.info("\n\n\n")
        
        chunk_text = self.concatenate_strings(chunk_text)
        return ["翻译全文",full_text, chunk_text, file_texts_path, file_name_md5_new, file_real_name, file_type, positions, page_no, chunk_id, mysql_id, file_from_folder, file_size, file_num_chars, file_upload_time, text_list, base64_list]
    
    