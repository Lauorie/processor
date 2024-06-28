import re
import json
import hashlib
from transformers import AutoTokenizer
import jieba
import os
from loguru import logger
import copy
from copy import deepcopy

class KnowledgeDocumentPreprocessor:
    
    def __init__(self,tokenizer_path, file_path, file_name):
        self.data = None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.file_path = file_path
        self.file_name = file_name
    
    
    # 1.预处理
    def preprocess(self):
        """
        1.去除header，footer，stage
        2.修改'SECTION-TITLE'为'SECTION-TEXT'
        3.所有非IMAGE类型的base64赋值“meaningless”
        4.所有TABLE类型的chunk添加一个text_list占位符
        5.处理page_no不唯一的错误情况
        """

        with open(self.file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        if self.data == []:
            logger.info(f"-------docai解析失败")
        self.temp_data = [chunk for chunk in self.data['tree'] if chunk['type']!='HEADER']
        self.data = [chunk for chunk in self.temp_data if chunk['type']!='FOOTER']
        # self.data = [chunk for chunk in self.data if not (chunk["display"]["top"] < (chunk["display"]["page_height"] / 20))]
        
        for chunk in self.data:
            if chunk['type'] == 'SECTION-TITLE':
                chunk['type'] = 'SECTION-TEXT'
            if chunk['type'] in ['SECTION-TEXT','TABLE', 'TITLE']:
                chunk['base64'] = 'meaningless'
            if chunk['type'] == 'TABLE':
                chunk['text_list'] = []
            chunk.pop('stage')
                
            page_no = copy.deepcopy(chunk['positions']['page_no'])
            if(len(page_no)>1): # 处理docai的错误类型
                    chunk['positions']['page_no'] = [page_no[0]]    
    
    
    # 2.识别title
    def is_title(self,text,i):
        """
        1.识别并标记title类型
        """
        title_keywords = (
            r'(^前.*言$|^简.*介$|目.*录$|^摘.*要$|^引.*言$|^结.*论$|^绪.*论$|^导.*言$|^背.*景$|^方.*法$|^结.*果$|^讨.*论$|^总.*结$|^附.*录$|^致.*谢$)',
            r'(^参考文献$|^版权声明$|^版权说明$)',
            r'(^第.*([一二三四五六七八九十百千万亿]|[1234567890]).*(章|节))',
            r'chapter.*[1234567890]$',
            r'(^preface|^abstract|^introduction|^catalogue|^experiment|^conclusion|^future work|^reference)'
        )
        if re.search(r'\.{6,}', text):
            return False
        if re.match(r'^\d', text.strip()):
            return False
         
        if i == 0:
            return True
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in title_keywords) and (self.calculate_text_len(text)<15):
            return True
        return False    
        
    # 通过分词器计算计算长度信息
    def calculate_length(self,text):
        return len(self.tokenizer.encode(text))
    
    
    # 合并子chunk
    # 1.跨IMAGE合并
    # 2.TABLE是一个完整的类
    def process_knowledge_chunks(self, chunk_list):
        result_chunks = []
        buffer_text = ""
        first_chunk = None
        display_list = []
        positions_list = []
        base64_list = []
        texts_list = []

        file_name_base = self.file_name
        extra_info = file_name_base + '的' + chunk_list[0]['text'] + '的部分内容为' + ':'

        for i, chunk in enumerate(chunk_list):
            try:
                if chunk['type'] == 'SECTION-TEXT':
                    temp_text = buffer_text + "\n" + chunk['text'] if buffer_text else chunk['text']
                    full_text = extra_info + temp_text
                    current_length = self.calculate_length(full_text)

                    if current_length < 500:
                        if not first_chunk:
                            first_chunk = chunk.copy()
                        buffer_text = temp_text
                        display_list.append(chunk.get('display', {}))
                        positions_list.append(chunk.get('positions', {}))
                        base64_list.append(chunk.get('base64', ''))
                        texts_list.append(chunk['text'])
                    else:
                        if first_chunk:
                            merged_chunk = first_chunk.copy()
                            merged_chunk['text'] = extra_info + buffer_text
                            merged_chunk['display'] = display_list[:]
                            merged_chunk['positions'] = positions_list[:]
                            merged_chunk['base64'] = base64_list[:]
                            merged_chunk['text_list'] = texts_list[:]
                            result_chunks.append(merged_chunk)

                        # Reset all lists and reinitialize from current chunk
                        first_chunk = chunk.copy()
                        buffer_text = chunk['text']
                        display_list = [chunk.get('display', {})]
                        positions_list = [chunk.get('positions', {})]
                        base64_list = [chunk.get('base64', '')]
                        texts_list = [chunk['text']]

                elif chunk['type'] == 'TABLE':
                    if buffer_text:
                        merged_chunk = first_chunk.copy()
                        merged_chunk['text'] = extra_info + buffer_text
                        merged_chunk['display'] = display_list[:]
                        merged_chunk['positions'] = positions_list[:]
                        merged_chunk['base64'] = base64_list[:]
                        merged_chunk['text_list'] = texts_list[:]
                        result_chunks.append(merged_chunk)
                        buffer_text = ""
                        display_list = []
                        positions_list = []
                        base64_list = []
                        texts_list = []
                    extra_table_info = file_name_base + '的' + chunk['text'] + '的一个表格为' + ':'
                    chunk['text'] = extra_table_info + chunk['text']
                    result_chunks.append(chunk)
                    first_chunk = None

                elif chunk['type'] == 'TITLE':
                    result_chunks.append(chunk)

            except Exception as e:
                print("Error processing chunk: ", e)

        if buffer_text:
            merged_chunk = first_chunk.copy()
            merged_chunk['text'] = extra_info + buffer_text
            merged_chunk['display'] = display_list[:]
            merged_chunk['positions'] = positions_list[:]
            merged_chunk['base64'] = base64_list[:]
            merged_chunk['text_list'] = texts_list[:]
            result_chunks.append(merged_chunk)

        for chunk in result_chunks:
            if isinstance(chunk['display'], dict):
                chunk['display'] = [chunk['display']]
            if isinstance(chunk['positions'], dict):
                chunk['positions'] = [chunk['positions']]
            if 'children' in chunk:
                del chunk['children']

        return result_chunks
    
    
    def process_long_context_chunks(self, chunk_list):
        result_chunks = []
        # first_chunk = None
        # current_page_no = None
        # display_list = [] 

        for chunk in chunk_list:
            if chunk['type'] == 'SECTION-TEXT':
                chunk['text'] = chunk['text']+'\n'
                result_chunks.append(chunk)
            else:
                # 非“SECTION-TEXT”类型，直接添加
                result_chunks.append(chunk)

        for chunk in result_chunks:
            if isinstance(chunk['display'], dict):
                chunk['display'] = [chunk['display']]
            if isinstance(chunk['positions'], dict):
                chunk['positions'] = [chunk['positions']]
            if 'children' in chunk:
                del chunk['children']

        return result_chunks
    
    # aux_func
    def rename_key(self,dictionary, old_key, new_key):
        """
        1.用于修改base64为base64_list
        """
        if old_key in dictionary:
            dictionary[new_key] = dictionary.pop(old_key)
        else:
            logger.info(f"Key '{old_key}' not found in the dictionary.")
          
            
    # aux_func
    def hash_md5(self,text):
        """
        1.生成hash编码
        """
        hash_object = hashlib.md5(text.encode())
        return hash_object.hexdigest()
    
    
    # aux_func
    def calculate_text_len(self, text):
        """
        1.计算长度
        """
        cleaned_text = re.sub(r'[\d!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text) 
        
        english_words = re.findall(r'[a-zA-Z]+', cleaned_text)
        english_word_count = len(english_words)
        
        chinese_characters = re.findall(r'[\u4e00-\u9fff]', cleaned_text)
        chinese_character_count = len(chinese_characters)
        
        total_length = english_word_count + chinese_character_count
        return total_length
    
    # aux_func
    def check_absolute_pos(self,chunk):
        """
        1.不是顶格
        """
        numeric_pattern = r'^[\d\-]*$'
        if re.match(numeric_pattern, chunk['text']):
            return False
        if (((chunk['display']['page_width']/20)*7) < chunk['display']['left']) and (chunk['display']['left']<((chunk['display']['page_width']/10)*5)):
            return True
    
    # aux_func
    def check_text_caption_type(self, input_string):
        numeric_pattern = r'^[\d\-]*$'
        if re.match(numeric_pattern, input_string):
            return False
        pattern = r'(?i)(图|figure|pic|image|图像|illustrate|illustration|show|demonstrate|展示|说明|图表|table|photo|照片|illustrated)(?:[:：\-—\s]*\d*)?'

        if not re.search(pattern, input_string):
            return False

        return True
    
    # axu_func
    def process_chunks_length(self, chunk_list):
        result_list = []
        for chunk in chunk_list:
            if chunk['type'] == 'SECTION-TEXT':
                text = chunk['text']
                words = list(jieba.cut(text))
                current_text = ""
                current_chunk = {}

                for word in words:
                    if self.calculate_text_len(current_text + word) <= 490:
                        current_text += word
                    else:
                        current_chunk = copy.deepcopy(chunk)
                        current_chunk['text'] = current_text
                        result_list.append(current_chunk)
                        current_text = word
                    
                if current_text:
                    current_chunk = copy.deepcopy(chunk)
                    current_chunk['text'] = current_text
                    result_list.append(current_chunk)
            else:
                result_list.append(chunk)
    
        return result_list
    
    
    # 主函数        
    def process(self):
        
        # 1. 预处理
        self.preprocess()
        
        # 2.强制把标题添加在第一个chunk，类型为title
        aux_chunk = {}
        aux_chunk['display'] = copy.deepcopy(self.data[0]['display'])
        aux_chunk['positions'] = copy.deepcopy(self.data[0]['positions'])
        aux_chunk['page_no'] = copy.deepcopy(self.data[0]['page_no'])
        aux_chunk['text'] = self.file_name
        aux_chunk['base64'] = 'meaningless'
        aux_chunk['type'] = 'TITLE'
        # aux_chunk['stage'] = copy.deepcopy(self.data[0]['stage'])
        self.data.insert(0, aux_chunk)
        
        # 3.识别title类型
        for i, chunk in enumerate(self.data):                
            if self.is_title(chunk['text'],i) and (chunk['type']=='SECTION-TEXT'):
                chunk['type'] = 'TITLE'
               
        # 4.根据Title类型标记索引
        index_list = []
        for i, chunk in enumerate(self.data):
            if chunk['type']=='TITLE':
                index_list.append(i)
         
        # 5.根据索引切分子list
        chunk_lists = []
        for i in range(len(index_list)):
            if (i+1)!=len(index_list):
                chunks = self.data[index_list[i]:index_list[i+1]]
            else:
                chunks = self.data[index_list[i]:]
            chunk_lists.append(chunks)
         
        # 6.根据识别到的title给每个chunk添加附加信息（用在Long_context部分）
        for chunk_list in chunk_lists:
            for i, chunk in enumerate(chunk_list):
                chunk['attach_text']=self.file_name + '的' + chunk_list[0]['text'] + '的部分内容为' + ':' + chunk['text']
        
        # 7.识别Caption
        for chunk_list in chunk_lists:
            for i, chunk in enumerate(chunk_list):
                if i > 0 and chunk_list[i-1]['type'] == 'IMAGE' and chunk['type'] == 'SECTION-TEXT' and (self.check_absolute_pos(chunk) or self.check_text_caption_type(chunk['text'])):
                    chunk['type'] ='CAPTION'
                if i < len(chunk_list) - 1 and chunk_list[i+1]['type'] == 'IMAGE' and chunk['type'] == 'SECTION-TEXT' and (self.check_absolute_pos(chunk) or self.check_text_caption_type(chunk['text'])):
                    chunk['type'] = 'CAPTION'
        
        # 8.IMAGE类型的text赋值为caption（后续应该用不到CAPTION类型）
        for chunk_list in chunk_lists:
            for i, chunk in enumerate(chunk_list):
                if chunk['type']=='IMAGE':
                    text_1 = ''
                    text_2 = ''
                    if i>0 and chunk_list[i-1]['type']=="CAPTION":
                        text_1 = chunk_list[i-1]['text']
                    if i < len(chunk_list) - 1 and chunk_list[i+1]['type']=='CAPTION':
                        text_2 = chunk_list[i+1]['text']
                    chunk['text'] = text_1+text_2
        
        # 9.修改IMAGE类型，当前只有TITLE，SECTION-TEXT(IMAGE合并进去了)，TABLE，CAPTION
        for chunk_list in chunk_lists:
            for chunk in chunk_list:
                if chunk['type'] == 'IMAGE':
                    chunk['type'] = 'SECTION-TEXT'
        
        # 10.处理长度超过512的文本（只针对section-text类型）
        chunk_lists = [self.process_chunks_length(chunk_list) for chunk_list in chunk_lists]
        
        # 11. 合并chunk
        chunk_knowledge_list = []
        for chunk_list in chunk_lists:
            chunk_knowledge_list.append(copy.deepcopy(self.process_knowledge_chunks(chunk_list)))
        
        chunk_long_context_list = []
        for chunk_list in chunk_lists:
            chunk_long_context_list.append(copy.deepcopy(self.process_long_context_chunks(chunk_list)))
        
        # 12. 给每个chunk添加id信息
        num = 0
        for chunk_list in chunk_knowledge_list:
            for chunk in chunk_list:
                if chunk['text'] is not None:
                    hash_id = self.hash_md5(chunk['text'])
                elif chunk['base64'] is not None:
                    hash_id = self.hash_md5(chunk['base64'])
                else:
                    hash_id = self.hash_md5('')
                id = str(num)+'-'+hash_id
                chunk['id']=id
                num+=1
        
        
        num = 0
        for chunk_list in chunk_long_context_list:
            for chunk in chunk_list:
                if chunk['text'] is not None:
                    hash_id = self.hash_md5(chunk['text'])
                elif chunk['base64'] is not None:
                    hash_id = self.hash_md5(chunk['base64'])
                else:
                    hash_id = self.hash_md5('')
                id = str(num)+'-'+hash_id
                chunk['id']=id
                num+=1
                
                
        # 13. 给底层chunk添加parent_id
        for chunk_list in chunk_knowledge_list:
            for i, chunk in enumerate(chunk_list):
                # if len(chunk_list)>1:
                parent_id = chunk_list[0]['id']
                chunk['page_no'] = chunk['page_no']+1
                if chunk['type'] == 'TITLE':
                    chunk['parent_id'] = parent_id
                if chunk['type'] in ('SECTION-TEXT','TABLE','IMAGE'):
                    chunk['parent_id'] = parent_id
                # if chunk['type']=='CAPTION':
                #     chunk['parent_id'] = chunk_list[i+chunk['parent_pos']]['id']
                #     del chunk['parent_pos']
                
                for cm, pos in enumerate(chunk['positions']):
                    if isinstance(pos, dict):
                        chunk['positions'][cm]['bbox'] = chunk['positions'][cm]['bbox'][0]
                        
        for j, chunk_list in enumerate(chunk_long_context_list):
            for i, chunk in enumerate(chunk_list):
                # if len(chunk_list)>1:
                parent_id = chunk_list[0]['id']
                chunk['page_no'] = chunk['page_no']+1
                if chunk['type'] == 'TITLE':
                    chunk['parent_id'] = parent_id
                if chunk['type'] in ('SECTION-TEXT','TABLE','IMAGE'):
                    chunk['parent_id'] = parent_id
                # if chunk['type']=='CAPTION':
                #     chunk['parent_id'] = chunk_list[i+chunk['parent_pos']]['id']
                #     del chunk['parent_pos']
                
                for cm, pos in enumerate(chunk['positions']):
                    if isinstance(pos, dict):
                        chunk['positions'][cm]['bbox'] = chunk['positions'][cm]['bbox'][0]
                # if j == 0 and i==0:
                #     chunk['positions'] = chunk_list[1]['positions']
                    
                    
        # 14. 返回列表类型
        final_knowledge_chunk_list = []    
        for chunk_list in chunk_knowledge_list:
            for chunk in chunk_list:
                if chunk['type']!='TITLE':
                    self.rename_key(chunk, 'base64', 'base64_list')
                    if chunk['type'] == 'TABLE':
                        chunk['base64_list'] = [chunk['base64_list']]
                        chunk['text_list'] = [chunk['text']]
                    final_knowledge_chunk_list.append(chunk)
        
        final_long_context_chunk_list = []    
        for chunk_list in chunk_long_context_list:
            for chunk in chunk_list:
                # if chunk['type']=='SECTION-TEXT' and chunk['base64'] != 'meaningless':
                #     chunk['type'] = 'IMAGE'
                if chunk['type']!='CAPTION':
                    self.rename_key(chunk, 'base64', 'base64_list')
                    chunk['text_list'] = chunk['text']
                    chunk['base64_list'] = chunk['base64_list']
                    final_long_context_chunk_list.append(chunk)

        return (final_knowledge_chunk_list, final_long_context_chunk_list)

if __name__ =="__main__":
    tokenizer_path = '/root/web_demo/HybirdSearch/models/models--Qwen--Qwen1.5-14B-Chat'
    file_path = '/root/web_demo/HybirdSearch/cmx_workapace/es_app_0625_index_type/unprocessed/0bd758fc16f4266de9e5f39e72905129.json'
    file_name = '0bd758fc16f4266de9e5f39e72905129'   # 别带.pdf
    dp = KnowledgeDocumentPreprocessor(tokenizer_path,  file_path, file_name)
    knowledge, long_content = dp.process()
    
    with open('/root/web_demo/HybirdSearch/cmx_workapace/es_app_0625_index_type/问答_processed/0bd758fc16f4266de9e5f39e72905129.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=4)
    with open('/root/web_demo/HybirdSearch/cmx_workapace/es_app_0625_index_type/长文本_processed/0bd758fc16f4266de9e5f39e72905129.json', 'w', encoding='utf-8') as f:
        json.dump(long_content, f, ensure_ascii=False, indent=4)