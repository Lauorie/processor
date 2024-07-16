import re
import json
import hashlib
from transformers import AutoTokenizer
import jieba
import os
from loguru import logger
import copy
from copy import deepcopy
import string

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
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
            if self.data == []:
                logger.info(f"-------docai解析失败")
            self.temp_data = [chunk for chunk in self.data['tree'] if chunk['type']!='HEADER']
            self.data = [chunk for chunk in self.temp_data if chunk['type']!='FOOTER']
        except:
            logger.info(f"第1步的header，footer处理异常")
        
        try:
            table_list = []
            remove_index = None
            for index, chunk in enumerate(self.data):
                chunk.pop('stage', None)
                if chunk['type'] == 'SECTION-TITLE':
                    chunk['type'] = 'SECTION-TEXT'
                if chunk['type'] in ['SECTION-TEXT', 'TABLE', 'TITLE']:
                    chunk['base64'] = 'meaningless'
                    
                if chunk['type'] == 'TABLE':
                    chunk['text_list'] = []
                    if chunk['text'][:11] == '\n\n\n问题\n答案\n\n\n':
                        chunk['type'] = 'FAQ'
                        try:
                            split_text = chunk['text'].split('\n\n\n')
                            if len(split_text) > 1 and split_text[1] == '问题\n答案':
                                remove_index = index
                                text_list = split_text[2:]
                                for iner_index, text in enumerate(text_list[:-1]):
                                    aux_chunk = copy.deepcopy(chunk)
                                    aux_chunk['text'] = '问题\t答案' + "\n" + text
                                    table_list.append(aux_chunk)
                        except Exception as e:
                            logger.error(f"Error processing TABLE chunk at index {index}: {str(e)}", exc_info=True)
                    # doc类型
                    elif chunk['text'].split('\n')[0]=='问题\t答案':
                        logger.info(f"表格为doc类型")
                        chunk['type'] = 'FAQ'
                        try:
                            remove_index = index
                            text = chunk['text'].split('\t', 1)[1]
                            
                            # 每段一答一问的列表
                            text_list = ['问题']
                            merged_list = []
                            split_text = text.split('\t') # 一个回答，一个问题
                            logger.info(f"split_text:{split_text[-1]}") # 最后一个是一个回答
                            
                            for ans_que_text in split_text[:-1]:
                                text_1 = ans_que_text.rsplit('\n', 1)[0]
                                text_2 = ans_que_text.rsplit('\n', 1)[1]
                                text_list.append(text_1)
                                text_list.append(text_2)
                            text_list.append(split_text[-1])

                            if len(text_list)%2==0:
                                logger.info(f"偶数个")
                                text_list = text_list[2:]
                            # logger.info(f"text_list:{text_list[-2:]}")
                            # 使用一个步长为2的循环来遍历text_list
                            for i in range(0, len(text_list), 2):
                                if i+1 < len(text_list):
                                    merged_list.append('问题\t答案'+'\n'+ text_list[i] + '\t' + text_list[i+1])
                                else:
                                    merged_list.append(text_list[i])
                            
                            for i,merged_text in enumerate(merged_list):
                                # logger.info(f"开始创建第{i}个aux_chunk")
                                aux_chunk = {
                                    "display": {
                                    "left": 0.0,
                                    "top": 676790.5464800077,
                                    "right": 483.6363636363637,
                                    "bottom": 0.0,
                                    "page_width": 483.6363636363637,
                                    "page_height": 676790.5464800077,
                                },
                                    "positions": {
                                        "bbox": [
                                        [
                                            0.0,
                                            676790.5464800077,
                                            483.6363636363637,
                                            0.0
                                        ]
                                        ],
                                        "page_height": 676790,
                                        "page_no": [
                                        0
                                        ],
                                        "page_width": 483
                                    },
                                    "page_no": [0],
                                    "text":merged_text,
                                    "base64": None,
                                    "type": "FAQ",
                                    "stage": "unkown",
                                    "children": None
                                }
                                
                                
                                # aux_chunk = copy.deepcopy(chunk)
                                # aux_chunk['text'] = merged_text
                                table_list.append(aux_chunk)
                            logger.info(f"chunk:{table_list[-1]['text']}")
                        except Exception as e:
                            logger.error(f"Error processing TABLE chunk at index {index}: {str(e)}", exc_info=True)
                    else:
                        if self.calculate_text_len(chunk['text']) > 8001:
                            logger.info(f"执行8000切片")
                            chunk['text'] = chunk['text'][:8000]
                        else:
                            pass
                        
            if remove_index is not None:
                logger.info(f"执行remove_index")
                self.data.pop(remove_index)
                for i, chunk in enumerate(table_list):
                    self.data.insert(remove_index + i, chunk)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            error_line = traceback.extract_tb(exc_tb)[-1].line
            logger.error(f"Exception in {fname} at line {line_number}: {str(e)}\nError occurred at line: {error_line}", exc_info=True)
                
        try:
            page_no = copy.deepcopy(chunk['positions']['page_no'])
            if(len(page_no)>1): # 处理docai的错误类型
                chunk['positions']['page_no'] = [page_no[0]]
        except:
            logger.info(f"第1步中，docai的page_no解析异常")    
    
    
    # 2.识别title
    def is_title_general(self,text,i):
        """
        1.识别并标记title类型
        """
        title_keywords = (
            r'(^前\s*言|^简\s*介|^目\s*录|^摘\s*要|^引\s*言|^结\s*论|^绪\s*论|^导\s*言|^背\s*景|^方\s*法|^结\s*果|^讨\s*论|^总\s*结|^附\s*录|^致\s*谢)',
            r'(^参\s*考\s*文\s*献|^版\s*权\s*声\s*明|^版\s*权\s*说\s*明)',
            r'(^第\s*([一二三四五六七八九十百千万亿]|[1234567890])\s*(章|节))',
        )
        title_keywords_with_len = (
            r'(前\s*言|简\s*介|目\s*录|摘\s*要|引\s*言|结\s*论|绪\s*论|导\s*言|背\s*景|结\s*果|讨\s*论|总\s*结|附\s*录|致\s*谢)',
            r'(参\s*考\s*文\s*献|版\s*权\s*声\s*明|版\s*权\s*说\s*明)',
            r'(第\s*([一二三四五六七八九十百千万亿]|[1234567890])\s*(章|节))',
        )
        title_keywords_en = (
            r'chapter.*[1234567890]$', 
            r'(^preface|^abstract|^introduction|^catalogue|^experiment|^conclusion|^future work|^reference)', 
            r'^[0-9]+ *(preface|abstract|introduction|catalogue|experiment|conclusion|future work|reference)'
        )
        
        if re.search(r'\.{6,}', text):
            return False
        if i == 0:
            return True
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in title_keywords_en):
            return True
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in title_keywords):
            return True
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in title_keywords_with_len) and self.calculate_text_len(text)<10 and self.starts_with_punctuation(text):
            return True
        # 1 Abstract
        # pattern_one = r'^\d+\s+[^.,?!:;\s]+'
        # if re.match(pattern_one, text) and self.check_string(text) and self.calculate_text_len(text)<15:
        #     return True
        return False    
    
    
    def is_title_special(self,text,i):
        """
        1.识别并标记title类型
        """
        if re.search(r'\.{6,}', text):
            return False
        # 二、摘要
        pattern_chin = r'^[一二三四五六七八九十百千万亿]+、\s*[^.,?!:;\s]+'
        if re.match(pattern_chin, text) and self.check_string(text) and self.calculate_text_len(text)<20 and self.less_than_half_digits(text):
            return True
        # 1、引言
        pattern_chin = r'^\d+、\s*[^.,?!:;\s]+'
        if re.match(pattern_chin, text) and self.check_string(text) and self.calculate_text_len(text)<20 and self.less_than_half_digits(text):
            return True
        # 1.2 实验
        pattern_num_dot = r'^\d+(\s*\.\s*\d+)+\s*[^.,?!:;\s]+'
        if re.match(pattern_num_dot, text) and self.check_string(text) and self.calculate_text_len(text)<20 and self.less_than_half_digits(text):
            return True
        # 1 相关工作
        pattern_num_char = r'^(\d+|[零一二三四五六七八九十百千万亿]+)\s+[\u4e00-\u9fa5a-zA-Z]+'
        if re.match(pattern_num_char, text) and self.check_string(text) and self.calculate_text_len(text)<15 and self.less_than_half_digits(text):
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

                elif chunk['type'] in ['TITLE','FAQ']:
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
            if chunk['type'] in ['TABLE','FAQ']:
                chunk['type'] = 'SECTION-TEXT'
            if chunk['type'] in ['SECTION-TEXT','TITLE','SECTION','SECTION-TITLE']:
                chunk['text'] = chunk['text']+'\n'
                result_chunks.append(chunk)
            elif chunk['type']!='IMAGE':
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
    def starts_with_punctuation(self, text):
        if not text.strip():
            return True
        if text[0] in string.punctuation:
            return False
        else:
            return True
    
    # aux_func
    def less_than_half_digits(self, text):
        digit_count = 0
        non_digit_count = 0

        for char in text:
            if char.isdigit():
                digit_count += 1
            else:
                non_digit_count += 1
        return digit_count < non_digit_count / 2
    
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
    def check_string(self, text):
        pattern = r'^[\d.,?!:;]+$'
        if re.match(pattern, text):
            return False
        return True  
            
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
        try:
            self.preprocess()
        except:
            logger.info(f"preprocesser的第1步出错")
        
        
        # 2.强制把标题添加在第一个chunk，类型为title
        try:
            aux_chunk = {}
            aux_chunk['display'] = copy.deepcopy(self.data[0]['display'])
            aux_chunk['positions'] = copy.deepcopy(self.data[0]['positions'])
            aux_chunk['page_no'] = copy.deepcopy(self.data[0]['page_no'])
            aux_chunk['text'] = self.file_name
            aux_chunk['attach_text'] = self.file_name
            aux_chunk['base64'] = 'meaningless'
            aux_chunk['type'] = 'TITLE'
            # aux_chunk['stage'] = copy.deepcopy(self.data[0]['stage'])
            self.data.insert(0, aux_chunk)
            knowledge_data, long_data = copy.deepcopy(self.data), copy.deepcopy(self.data)
        except:
            logger.info(f"preprocesser的第2步出错")
        
        
        # 3.通过关键字和数字开头识别title类型，并将其添加到attach_text，
        try:
            for i, chunk in enumerate(knowledge_data):                
                if self.is_title_general(chunk['text'],i) and (chunk['type']=='SECTION-TEXT'):
                    chunk['type'] = 'TITLE'
                    if self.calculate_text_len(chunk['text'])<10:
                        chunk['attach_text'] = chunk['text']
                    else:
                        chunk['attach_text'] = chunk['text'][:10]
            
            for i, chunk in enumerate(long_data):                
                if self.is_title_general(chunk['text'],i) and (chunk['type']=='SECTION-TEXT'):
                    chunk['type'] = 'TITLE'
                    if self.calculate_text_len(chunk['text'])<10:
                        chunk['attach_text'] = chunk['text']
                    else:
                        chunk['attach_text'] = chunk['text'][:10]
                if self.is_title_special(chunk['text'],i) and (chunk['type']=='SECTION-TEXT'):
                    chunk['type'] = 'SECTION-TITLE'
                    if self.calculate_text_len(chunk['text'])<10:
                        chunk['attach_text'] = chunk['text']
                    else:
                        chunk['attach_text'] = chunk['text'][:10]
        except:
            logger.info(f"preprocesser的第3步出错")
        
        # 4.根据Title类型标记索引
        try:
            knowledge_index_list = []
            for i, chunk in enumerate(knowledge_data):
                if chunk['type']=='TITLE':
                    knowledge_index_list.append(i)
            
            long_index_list = []
            for i, chunk in enumerate(long_data):
                if chunk['type'] == 'TITLE':
                    long_index_list.append(i)
                elif chunk['type'] == 'SECTION-TITLE':
                    long_index_list.append(i)
        except:
            logger.info(f"preprocesser的第4步出错")
        
        # 5.根据索引切分子list
        try:
            knowledge_chunk_lists = []
            for i in range(len(knowledge_index_list)):
                if (i+1)!=len(knowledge_index_list):
                    chunks = knowledge_data[knowledge_index_list[i]:knowledge_index_list[i+1]]
                else:
                    chunks = knowledge_data[knowledge_index_list[i]:]
                knowledge_chunk_lists.append(chunks)
            
            long_chunk_lists = []
            for i in range(len(long_index_list)):
                if (i+1)!=len(long_index_list):
                    chunks = long_data[long_index_list[i]:long_index_list[i+1]]
                else:
                    chunks = long_data[long_index_list[i]:]
                long_chunk_lists.append(chunks)
        except:
            logger.info(f"preprocesser的第5步出错")
            
            
        # 6.根据识别到的title给每个chunk添加附加信息（用在Long_context部分）
        try:
            for chunk_list in knowledge_chunk_lists:
                for i, chunk in enumerate(chunk_list):
                    if i!=0:
                        chunk['attach_text']=self.file_name + '的' + chunk_list[0]['text'] + '的部分内容为' + ':' + chunk['text']
                        
                        
            for chunk_list in long_chunk_lists:
                for i, chunk in enumerate(chunk_list):
                    if i!=0:
                        chunk['attach_text']=self.file_name + '的' + chunk_list[0]['text'] + '的部分内容为' + ':' + chunk['text']
                        
        except:
            logger.info(f"preprocesser的第6步出错")
        
        # 7.识别Caption
        try:
            for chunk_list in knowledge_chunk_lists:
                for i, chunk in enumerate(chunk_list):
                    if i > 0 and chunk_list[i-1]['type'] == 'IMAGE' and chunk['type'] == 'SECTION-TEXT':
                        chunk['type'] ='CAPTION'
                    if i < len(chunk_list) - 1 and chunk_list[i+1]['type'] == 'IMAGE' and chunk['type'] == 'SECTION-TEXT':
                        chunk['type'] = 'CAPTION'
            
            for chunk_list in long_chunk_lists:
                for i, chunk in enumerate(chunk_list):
                    if i > 0 and chunk_list[i-1]['type'] == 'IMAGE' and chunk['type'] == 'SECTION-TEXT':
                        chunk['type'] ='CAPTION'
                    if i < len(chunk_list) - 1 and chunk_list[i+1]['type'] == 'IMAGE' and chunk['type'] == 'SECTION-TEXT':
                        chunk['type'] = 'CAPTION'
        except:
            logger.info(f"preprocesser的第7步出错")
        
        # 8.IMAGE类型的text赋值为caption（后续应该用不到CAPTION类型）
        try:
            for chunk_list in knowledge_chunk_lists:
                for i, chunk in enumerate(chunk_list):
                    if chunk['type']=='IMAGE':
                        text_1 = ''
                        text_2 = ''
                        text_self = ''
                        if i>0 and chunk_list[i-1]['type']=="CAPTION":
                            text_1 = chunk_list[i-1]['text']
                        if i < len(chunk_list) - 1 and chunk_list[i+1]['type']=='CAPTION':
                            text_2 = chunk_list[i+1]['text']
                        chunk['text'] = text_1+text_self+text_2
            
            for chunk_list in long_chunk_lists:
                for i, chunk in enumerate(chunk_list):
                    if chunk['type']=='IMAGE':
                        text_1 = ''
                        text_2 = ''
                        if i>0 and chunk_list[i-1]['type']=="CAPTION":
                            text_1 = chunk_list[i-1]['text']
                        if i < len(chunk_list) - 1 and chunk_list[i+1]['type']=='CAPTION':
                            text_2 = chunk_list[i+1]['text']
                        chunk['text'] = text_1+text_2
        except:
            logger.info(f"preprocesser的第8步出错")
        
        # 9.修改IMAGE类型，当前只有TITLE，SECTION-TEXT(IMAGE合并进去了)，TABLE，CAPTION
        try:
            for chunk_list in knowledge_chunk_lists:
                for chunk in chunk_list:
                    if chunk['type'] == 'IMAGE':
                        chunk['type'] = 'SECTION-TEXT'
            
            for chunk_list in long_chunk_lists:
                for chunk in chunk_list:
                    if chunk['type'] == 'IMAGE':
                        chunk['type'] = 'SECTION-TEXT'
        except:
            logger.info(f"preprocesser的第9步出错")
        
        # 10.处理长度超过512的文本（只针对section-text类型）
        try:
            knowledge_chunk_lists = [self.process_chunks_length(chunk_list) for chunk_list in knowledge_chunk_lists]
            long_chunk_lists = [self.process_chunks_length(chunk_list) for chunk_list in long_chunk_lists]
        except:
            logger.info(f"preprocesser的第10步出错")
        
        # 11. 合并chunk
        try:
            chunk_knowledge_list = []
            for chunk_list in knowledge_chunk_lists:
                chunk_knowledge_list.append(copy.deepcopy(self.process_knowledge_chunks(chunk_list)))
            
            chunk_long_context_list = []
            for chunk_list in long_chunk_lists:
                chunk_long_context_list.append(copy.deepcopy(self.process_long_context_chunks(chunk_list)))
        except:
            logger.info(f"preprocesser的第11步出错")
        
        # 12. 给每个chunk添加id信息
        try:
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
        except:
            logger.info(f"preprocesser的第12步出错")
                
                
        # 13. 给底层chunk添加parent_id
        try:
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
                    if chunk['type'] in ['TITLE','SECTION-TITLE']:
                        chunk['parent_id'] = parent_id
                    # if chunk['type'] in ('SECTION-TEXT','TABLE','IMAGE'):
                    # image和caption应该已经没有了
                    if chunk['type'] in ('SECTION-TEXT','TABLE'):
                        chunk['parent_id'] = parent_id
                    # if chunk['type']=='CAPTION':
                    #     chunk['parent_id'] = chunk_list[i+chunk['parent_pos']]['id']
                    #     del chunk['parent_pos']
                    
                    for cm, pos in enumerate(chunk['positions']):
                        if isinstance(pos, dict):
                            chunk['positions'][cm]['bbox'] = chunk['positions'][cm]['bbox'][0]
                    # if j == 0 and i==0:
                    #     chunk['positions'] = chunk_list[1]['positions']
        except:
            logger.info(f"preprocesser的第13步出错")
                    
                    
        # 14. 返回列表类型
        try:
            final_knowledge_chunk_list = []    
            for chunk_list in chunk_knowledge_list:
                for chunk in chunk_list:
                    chunk.pop('stage', None)
                    if chunk['type'] == 'FAQ':
                        chunk['base64_list'] = ['meaningless']
                        chunk['text_list'] = [chunk['text']]
                    if chunk['type']!='TITLE':
                        self.rename_key(chunk, 'base64', 'base64_list')
                        if chunk['type'] in ['TABLE','FAQ']:
                            chunk['base64_list'] = [chunk['base64_list']]
                            chunk['text_list'] = [chunk['text']]
                        if chunk['type'] == 'FAQ':
                            chunk["base64_list"] = ['meaningless']
                        final_knowledge_chunk_list.append(chunk)
            
            final_long_context_chunk_list = []    
            for chunk_list in chunk_long_context_list:
                for chunk in chunk_list:
                    chunk.pop('stage', None)
                    # if chunk['type']=='SECTION-TEXT' and chunk['base64'] != 'meaningless':
                    #     chunk['type'] = 'IMAGE'
                    if chunk['type']!='CAPTION':
                        self.rename_key(chunk, 'base64', 'base64_list')
                        chunk['text_list'] = chunk['text']
                        chunk['base64_list'] = chunk['base64_list']
                        final_long_context_chunk_list.append(chunk)
        except:
            logger.info(f"preprocesser的第14步出错")

        return (final_knowledge_chunk_list, final_long_context_chunk_list)

if __name__ =="__main__":
    tokenizer_path = '/root/web_demo/HybirdSearch/models/models--Qwen--Qwen1.5-14B-Chat'
    file_path = '/root/web_demo/HybirdSearch/cmx_workapace/es_app_0625_index_type/FAQ/5747bed03bbdc96a7751eb6e54926ecf.json'
    file_name = '5747bed03bbdc96a7751eb6e54926ecf'   # 别带.pdf
    dp = KnowledgeDocumentPreprocessor(tokenizer_path,  file_path, file_name)
    knowledge, long_content = dp.process()
    
    with open('/root/web_demo/HybirdSearch/cmx_workapace/es_app_0625_index_type/FAQprocessed/5747bed03bbdc96a7751eb6e54926ecf.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=4)
    # with open('/root/web_demo/HybirdSearch/cmx_workapace/es_app_0625_index_type/FAQprocessed/5747bed03bbdc96a7751eb6e54926ecf.json', 'w', encoding='utf-8') as f:
    #     json.dump(long_content, f, ensure_ascii=False, indent=4)