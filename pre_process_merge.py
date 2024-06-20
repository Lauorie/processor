# # 问题
# # 1.(sol)header中的章节信息去除不掉（例如，2022 年半年度报告.pdf(1).json）
# # 2.体现在哪里并翻译（目前是一个原始chunk，能否前后各拼接一个）
# # 3.bm25排序方法有bug
# # 4.翻译FA-AUTE这篇文章 被路由到翻译章节
# # 5.'(总结下面这段话)'被路由到总结title
# # 6.1.5 产品文档目录这种情况目录被误判
# # 翻译Safety FA-Aute的版本说明被路由到直接翻译


# # 删除display










# import re
# import json
# import hashlib
# from transformers import AutoTokenizer
# import jieba
# import os
# from loguru import logger
# import copy
# from copy import deepcopy

# class KnowledgeDocumentPreprocessor:
    
#     def __init__(self,tokenizer_path,  file_path, file_name):
#         self.data = None
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
#         self.file_path = file_path
#         self.file_name = file_name
    
#     # 处理chunk长度
#     def process_chunks_length(self, chunk_list):
#         result_list = []
#         for chunk in chunk_list:
#             if chunk['type'] == 'SECTION-TEXT':
#                 text = chunk['text']
#                 words = list(jieba.cut(text))  # 使用jieba进行中文分词
#                 current_text = ""
#                 current_chunk = {}

#                 for word in words:
#                     # 检查加上新词后是否超过500字符
#                     if self.calculate_text_len(current_text + word) <= 490:
#                         current_text += word
#                     else:
#                         # 保存当前chunk
#                         current_chunk = copy.deepcopy(chunk)
#                         current_chunk['text'] = current_text
#                         result_list.append(current_chunk)
#                         current_text = word  # 开始新的chunk
                    
#                 # 处理剩余的文本
#                 if current_text:
#                     current_chunk = copy.deepcopy(chunk)
#                     current_chunk['text'] = current_text
#                     result_list.append(current_chunk)
#             else:
#                 result_list.append(chunk)
    
#         return result_list
        
#     # 1 预处理，去除HEADER和FOOTER，调整'SECTION-TITLE'类型
#     def preprocess(self):
#         with open(self.file_path, 'r', encoding='utf-8') as file:
#             self.data = json.load(file)
#         if self.data == []:
#             logger.info(f"docai解析失败")
#         self.temp_data = [chunk for chunk in self.data['tree'] if chunk['type']!='HEADER']
#         self.data = [chunk for chunk in self.temp_data if chunk['type']!='FOOTER']
#         self.data = [chunk for chunk in self.data if not (chunk["display"]["top"] < (chunk["display"]["page_height"] / 20))]
#         for chunk in self.data:
#             if chunk['type'] == 'SECTION-TITLE':
#                 chunk['type'] = 'SECTION-TEXT'
            
                
                
#     # 计算字符串长度辅助判断是不是标题
#     def calculate_text_len(self, text):
#         cleaned_text = re.sub(r'[\d!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text) #substitute
        
#         # 计算英文长度
#         english_words = re.findall(r'[a-zA-Z]+', cleaned_text)
#         english_word_count = len(english_words)
        
#         # 计算中文长度
#         chinese_characters = re.findall(r'[\u4e00-\u9fff]', cleaned_text)
#         chinese_character_count = len(chinese_characters)
        
#         total_length = english_word_count + chinese_character_count
#         return total_length
    
    
#     # 2 判断是不是标题
#     def is_title(self,text,i):
#         title_keywords = (
#             r'(^前.*言$|^简.*介$|目.*录$|^摘.*要$|^引.*言$|^结.*论$|^绪.*论$|^导.*言$|^背.*景$|^方.*法$|^结.*果$|^讨.*论$|^总.*结$|^附.*录$|^致.*谢$)',
#             r'(^参考文献$|^版权声明$|^版权说明$)',
#             r'(^第.*([一二三四五六七八九十百千万亿]|[1234567890]).*(章|节))',
#             r'chapter.*[1234567890]$',
#             r'(^preface|^abstract|^introduction|^catalogue|^experiment|^conclusion|^future work|^reference)'
#         )
#         if re.search(r'\.{6,}', text):
#             return False
#         if re.match(r'^\d', text.strip()):
#             return False
         
#         if i == 0:
#             return True
#         elif any(re.search(pattern, text, re.IGNORECASE) for pattern in title_keywords) and (self.calculate_text_len(text)<15):
#             return True
#         return False
    
    
#     # 检查长度和关键字辅助判断caption类型
#     def check_text_caption_type(self,input_string):
#         if not re.search(r'图|Figure|figure', input_string):
#             return False
#         # length = self.calculate_text_len(input_string)
#         return True
    
    
#     # 查看文字的布局信息辅助辅助判断caption类型
#     def check_absolute_pos(self,chunk):
#         if (((chunk['display']['page_width']/10)*3) < chunk['display']['left']) and (chunk['display']['left']<((chunk['display']['page_width']/10)*5)):
#             return True
        
        
#     # 通过分词器计算计算长度信息
#     def calculate_length(self,text):
#         return len(self.tokenizer.encode(text))
    
    
#     # 合并子chunk
#     # 1.跨IMAGE合并
#     # 2.TABLE是一个完整的类
    # def process_knowledge_chunks(self, chunk_list):
    #     result_chunks = []
    #     buffer_text = ""
    #     first_chunk = None
    #     display_list = []  # 使用列表来存储所有相关的 display 信息
    #     positions_list = []  # 使用列表来存储所有相关的 position 信息

    #     file_name_base = self.file_name
    #     extra_info = file_name_base + '的' + chunk_list[0]['text'] + '的部分内容为' + ':'
    #     extra_table_info = file_name_base + '的' + chunk_list[0]['text'] + '的一个表格为' + ':'

    #     for i, chunk in enumerate(chunk_list):
    #         try:
    #             if chunk['type'] == 'SECTION-TEXT':
    #                 if not first_chunk:
    #                     first_chunk = chunk.copy()
    #                     display_list = [chunk.get('display', {})]
    #                     positions_list = [chunk.get('positions', {})]
    #                 else:
    #                     display_list.append(chunk.get('display', {}))
    #                     positions_list.append(chunk.get('positions', {}))

    #                 buffer_text += "\n" + chunk['text']
    #                 full_text = extra_info + buffer_text
    #                 current_length = self.calculate_length(full_text)

    #                 if current_length < 500:
    #                     continue

    #                 while current_length > 500:
    #                     temp_text = ''
    #                     for word in re.split(r'\s+', buffer_text):
    #                         if self.calculate_length(extra_info + temp_text + ' ' + word) <= 500:
    #                             temp_text += ' ' + word
    #                         else:
    #                             break
    #                     merged_chunk = first_chunk.copy()
    #                     merged_chunk['text'] = extra_info + temp_text.strip()
    #                     merged_chunk['display'] = display_list[:]
    #                     merged_chunk['positions'] = positions_list[:]
    #                     result_chunks.append(merged_chunk)
    #                     buffer_text = buffer_text[len(temp_text.strip()):].strip()
    #                     full_text = extra_info + buffer_text
    #                     current_length = self.calculate_length(full_text)
    #                     display_list = []  # 重置display列表
    #                     positions_list = []  # 重置positions列表

    #             elif chunk['type'] == 'TABLE':
    #                 # 处理TABLE类型，添加额外信息并清空当前合并缓存，直接添加TABLE chunk
    #                 if buffer_text:
    #                     merged_chunk = first_chunk.copy()
    #                     merged_chunk['text'] = extra_info + buffer_text
    #                     merged_chunk['display'] = display_list[:]
    #                     merged_chunk['positions'] = positions_list[:]
    #                     result_chunks.append(merged_chunk)
    #                     buffer_text = ""
    #                     display_list = []  # 清空display列表
    #                     positions_list = []  # 清空positions列表
    #                 chunk['text'] = extra_table_info + chunk['text']
    #                 result_chunks.append(chunk)
    #                 first_chunk = None  # 重新初始化合并过程

    #             elif chunk['type'] == 'TITLE':
    #                 # 对TITLE类型的处理，直接添加到结果列表中
    #                 result_chunks.append(chunk)

    #             else:
    #                 # 对IMAGE和CAPTION类型的处理
    #                 if buffer_text and (chunk['type'] == 'IMAGE' or chunk['type'] == 'CAPTION'):
    #                     display_list.append(chunk.get('display', {}))
    #                     positions_list.append(chunk.get('positions', {}))

    #         except Exception as e:
    #             print("Error processing chunk: ", e)

    #     if buffer_text:
    #         merged_chunk = first_chunk.copy()
    #         merged_chunk['text'] = extra_info + buffer_text
    #         merged_chunk['display'] = display_list[:]
    #         merged_chunk['positions'] = positions_list[:]
    #         result_chunks.append(merged_chunk)

    #     for chunk in result_chunks:
    #         if isinstance(chunk['display'], dict):
    #             chunk['display'] = [chunk['display']]
    #         if isinstance(chunk['positions'], dict):
    #             chunk['positions'] = [chunk['positions']]
    #         if 'children' in chunk:
    #             del chunk['children']

    #     return result_chunks




    
    
#     def process_long_context_chunks(self, chunk_list):
#         result_chunks = []
#         buffer_text = ""
#         first_chunk = None
#         current_page_no = None
#         display_list = []  # 使用列表来存储所有相关的 display 信息
#         positions_list = []  # 使用列表来存储所有相关的 position 信息

#         for chunk in chunk_list:
#             try:
#                 if chunk['type'] == 'SECTION-TEXT':
#                     if self.calculate_length(chunk['text']) < 500:
#                         # 如果长度小于500，直接添加到结果列表
#                         chunk['text'] = chunk['text']+'\n'
#                         result_chunks.append(chunk)
#                     else:
#                         # 长度超过500，进行拆分处理
#                         buffer_text = chunk['text']
#                         extra_info = ''
#                         current_length = self.calculate_length(buffer_text)
#                         while current_length > 500:
#                             temp_text = ''
#                             for word in re.split(r'\s+', buffer_text):
#                                 if self.calculate_length(extra_info + temp_text + ' ' + word) <= 500:
#                                     temp_text += ' ' + word
#                                 else:
#                                     break
#                             merged_chunk = copy.deepcopy(chunk)
#                             merged_chunk['text'] = extra_info + temp_text.strip()
#                             result_chunks.append(merged_chunk)
#                             buffer_text = buffer_text[len(temp_text.strip()):].strip()
#                             current_length = self.calculate_length(extra_info + buffer_text)
#                         if buffer_text:
#                             # 处理剩余部分
#                             merged_chunk = copy.deepcopy(chunk)
#                             merged_chunk['text'] = extra_info + buffer_text
#                             result_chunks.append(merged_chunk)
#                 else:
#                     # 非“SECTION-TEXT”类型，直接添加
#                     result_chunks.append(chunk)

#             except Exception as e:
#                 print("Error processing chunk: ", e)
            
#             for chunk in result_chunks:
#                 if isinstance(chunk['display'], dict):
#                     chunk['display'] = [chunk['display']]
#                 if isinstance(chunk['positions'], dict):
#                     chunk['positions'] = [chunk['positions']]
#                 if 'children' in chunk:
#                     del chunk['children']

#         return result_chunks
    


#     # 哈希编码
#     def hash_md5(self,text):
#         hash_object = hashlib.md5(text.encode())
#         return hash_object.hexdigest()
    
    
#     # 主函数        
#     def process(self):
        
#         # 1. 预处理（去除header，footer）
#         self.preprocess()
#         # 2. 判断Title类型（aux强制把标题添加在第一个chunk，类型为title）
#         aux_chunk = {}
#         aux_chunk['display'] = copy.deepcopy(self.data[0]['display'])
#         aux_chunk['positions'] = copy.deepcopy(self.data[0]['positions'])
#         aux_chunk['page_no'] = copy.deepcopy(self.data[0]['page_no'])
#         aux_chunk['text'] = self.file_name
#         aux_chunk['base64'] = None
#         aux_chunk['type'] = 'TITLE'
#         aux_chunk['stage'] = copy.deepcopy(self.data[0]['stage'])
#         # 统一在开头位置插入一个text为file_name的chunk
#         self.data.insert(0, aux_chunk)
#         for i, chunk in enumerate(self.data):                
#             if self.is_title(chunk['text'],i) and (chunk['type']=='SECTION-TEXT'):
#                 chunk['type'] = 'TITLE'
        
        
#         # 3. 根据Title类型标记索引
#         index_list = []
#         for i, chunk in enumerate(self.data):
#             if chunk['type']=='TITLE':
#                 index_list.append(i)
            
                
#         # 4. 根据索引切分子list
#         chunk_lists = []
#         for i in range(len(index_list)):
#             if (i+1)!=len(index_list):
#                 chunks = self.data[index_list[i]:index_list[i+1]]
#             else:
#                 chunks = self.data[index_list[i]:]
#             chunk_lists.append(chunks)
            
            
#         # 5. 给每个chunk添加附加信息
#         for chunk_list in chunk_lists:
#             for i, chunk in enumerate(chunk_list):
#                 chunk['attach_text']=self.file_name + '的' + chunk_list[0]['text'] + '的部分内容为' + ':' + chunk['text']
#                 page_no = copy.deepcopy(chunk['positions']['page_no'])
#                 if(len(page_no)>1):
#                     chunk['positions']['page_no'] = [page_no[0]]
                
                
#         # 6. 判断type为caption的chunk
#         for chunk_list in chunk_lists:
#             for i, chunk in enumerate(chunk_list):
#                 if (chunk['type']=='CAPTION'):
#                     chunk['parent_pos'] = -1
#                 if (chunk['type']=='IMAGE'):
#                     if i!=0 and i!=len(chunk_list)-1:
#                         if (chunk_list[i+1]['type']=='SECTION-TEXT' or chunk_list[i+1]['type']=='CAPTION')and self.check_text_caption_type(chunk_list[i+1]['text']):
#                             chunk_list[i+1]['type'] = 'CAPTION'
#                             chunk_list[i+1]['parent_pos'] = 1
#                             chunk_list[i+1]['state'] = 'USED'
#                         elif (chunk_list[i-1]['type']=='SECTION-TEXT' or chunk_list[i-1]['type']=='CAPTION')and  self.check_text_caption_type(chunk_list[i-1]['text']) and ('state' in chunk_list[i-1]):
#                             chunk_list[i-1]['type'] = 'CAPTION'
#                             chunk_list[i-1]['parent_pos'] = -1
#         chunk_a = chunk_lists
#         chunk_lists = []
#         for chunk_list in chunk_a:
#             chunk_lists.append(self.process_chunks_length(chunk_list))
        
        
        
#         # 7. 合并chunk
#         chunk_knowledge_list = []
#         for chunk_list in chunk_lists:
#             chunk_knowledge_list.append(copy.deepcopy(self.process_knowledge_chunks(chunk_list)))
        
#         chunk_long_context_list = []
#         for chunk_list in chunk_lists:
#             chunk_long_context_list.append(copy.deepcopy(self.process_long_context_chunks(chunk_list)))
            
            
#         # 8. 给每个chunk添加id信息，并删除stage
#         num = 0
#         for chunk_list in chunk_knowledge_list:
#             for chunk in chunk_list:
#                 if chunk['text'] is not None:
#                     hash_id = self.hash_md5(chunk['text'])
#                 elif chunk['base64'] is not None:
#                     hash_id = self.hash_md5(chunk['base64'])
#                 else:
#                     hash_id = self.hash_md5('')
#                 id = str(num)+'-'+hash_id
#                 chunk['id']=id
#                 chunk.pop('stage')
#                 num+=1
        
        
#         num = 0
#         for chunk_list in chunk_long_context_list:
#             for chunk in chunk_list:
#                 if chunk['text'] is not None:
#                     hash_id = self.hash_md5(chunk['text'])
#                 elif chunk['base64'] is not None:
#                     hash_id = self.hash_md5(chunk['base64'])
#                 else:
#                     hash_id = self.hash_md5('')
#                 id = str(num)+'-'+hash_id
#                 chunk['id']=id
#                 chunk.pop('stage')
#                 num+=1
                
                
#         # 9. 给底层chunk添加parent_id
#         for chunk_list in chunk_knowledge_list:
#             for i, chunk in enumerate(chunk_list):
#                 # if len(chunk_list)>1:
#                 parent_id = chunk_list[0]['id']
#                 chunk['page_no'] = chunk['page_no']+1
#                 if chunk['type'] == 'TITLE':
#                     chunk['parent_id'] = parent_id
#                 if chunk['type'] in ('SECTION-TEXT','TABLE','IMAGE'):
#                     chunk['parent_id'] = parent_id
#                 if chunk['type']=='CAPTION':
#                     chunk['parent_id'] = chunk_list[i+chunk['parent_pos']]['id']
#                     del chunk['parent_pos']
                
#                 for cm, pos in enumerate(chunk['positions']):
#                     if isinstance(pos, dict):
#                         chunk['positions'][cm]['bbox'] = chunk['positions'][cm]['bbox'][0]
                        
                
                
                    
                        
                
                    
                    
#         for j, chunk_list in enumerate(chunk_long_context_list):
#             for i, chunk in enumerate(chunk_list):
#                 # if len(chunk_list)>1:
#                 parent_id = chunk_list[0]['id']
#                 chunk['page_no'] = chunk['page_no']+1
#                 if chunk['type'] == 'TITLE':
#                     chunk['parent_id'] = parent_id
#                 if chunk['type'] in ('SECTION-TEXT','TABLE','IMAGE'):
#                     chunk['parent_id'] = parent_id
#                 if chunk['type']=='CAPTION':
#                     chunk['parent_id'] = chunk_list[i+chunk['parent_pos']]['id']
#                     del chunk['parent_pos']
                
#                 for cm, pos in enumerate(chunk['positions']):
#                     if isinstance(pos, dict):
                        
#                         chunk['positions'][cm]['bbox'] = chunk['positions'][cm]['bbox'][0]
                
                
                
#                 if j == 0 and i==0:
#                     chunk['positions'] = chunk_list[1]['positions']
                    
                    
#         # 10. 返回列表类型
#         final_knowledge_chunk_list = []    
#         for chunk_list in chunk_knowledge_list:
#             for chunk in chunk_list:
#                 if chunk['type']!='TITLE':
#                     final_knowledge_chunk_list.append(chunk)
#                 # final_knowledge_chunk_list.append(chunk)
        
#         final_long_context_chunk_list = []    
#         for chunk_list in chunk_long_context_list:
#             for chunk in chunk_list:
#                 final_long_context_chunk_list.append(chunk)

#         return (final_knowledge_chunk_list, final_long_context_chunk_list)

# if __name__ =="__main__":
#     tokenizer_path = '/root/web_demo/HybirdSearch/models/models--Qwen--Qwen1.5-14B-Chat'
#     file_path = '/root/web_demo/HybirdSearch/cmx_workapace/es_app_0613/unprocessed/2022 年半年度报告.pdf.json'
#     file_name = '2022 年半年度报告'   # 别带.pdf
#     dp = KnowledgeDocumentPreprocessor(tokenizer_path,  file_path, file_name)
#     knowledge, long_content = dp.process()
    
#     with open('/root/web_demo/HybirdSearch/cmx_workapace/es_app_0613/问答_processed/2022 年半年度报告.json', 'w', encoding='utf-8') as f:
#         json.dump(knowledge, f, ensure_ascii=False, indent=4)
#     with open('/root/web_demo/HybirdSearch/cmx_workapace/es_app_0613/长文本_processed/2022 年半年度报告.json', 'w', encoding='utf-8') as f:
#         json.dump(long_content, f, ensure_ascii=False, indent=4)



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
    
    # 处理chunk长度
    def process_chunks_length(self, chunk_list):
        result_list = []
        for chunk in chunk_list:
            if chunk['type'] == 'SECTION-TEXT':
                text = chunk['text']
                words = list(jieba.cut(text))  # 使用jieba进行中文分词
                current_text = ""
                current_chunk = {}

                for word in words:
                    # 检查加上新词后是否超过500字符
                    if self.calculate_text_len(current_text + word) <= 490:
                        current_text += word
                    else:
                        # 保存当前chunk
                        current_chunk = copy.deepcopy(chunk)
                        current_chunk['text'] = current_text
                        result_list.append(current_chunk)
                        current_text = word  # 开始新的chunk
                    
                # 处理剩余的文本
                if current_text:
                    current_chunk = copy.deepcopy(chunk)
                    current_chunk['text'] = current_text
                    result_list.append(current_chunk)
            else:
                result_list.append(chunk)
    
        return result_list
        
    # 1 预处理，去除HEADER和FOOTER，调整'SECTION-TITLE'类型，给所有非IMAGE类型的base64赋值“meaningless”,所有IMAGE类型的chunk添加一个占位符
    def preprocess(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        if self.data == []:
            logger.info(f"docai解析失败")
        self.temp_data = [chunk for chunk in self.data['tree'] if chunk['type']!='HEADER']
        self.data = [chunk for chunk in self.temp_data if chunk['type']!='FOOTER']
        self.data = [chunk for chunk in self.data if not (chunk["display"]["top"] < (chunk["display"]["page_height"] / 20))]
        for chunk in self.data:
            if chunk['type'] == 'SECTION-TITLE':
                chunk['type'] = 'SECTION-TEXT'
            # if chunk['type'] == 'IMAGE':
            #     chunk['text'] = '![image](attachment:image)'
            if chunk['type'] in ['SECTION-TEXT','TABLE', 'TITLE']:
                chunk['base64'] = 'meaningless'
        
    def rename_key(self,dictionary, old_key, new_key):
        if old_key in dictionary:
            dictionary[new_key] = dictionary.pop(old_key)
        else:
            print(f"Key '{old_key}' not found in the dictionary.")
                
    # 计算字符串长度辅助判断是不是标题
    def calculate_text_len(self, text):
        cleaned_text = re.sub(r'[\d!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text) #substitute
        
        # 计算英文长度
        english_words = re.findall(r'[a-zA-Z]+', cleaned_text)
        english_word_count = len(english_words)
        
        # 计算中文长度
        chinese_characters = re.findall(r'[\u4e00-\u9fff]', cleaned_text)
        chinese_character_count = len(chinese_characters)
        
        total_length = english_word_count + chinese_character_count
        return total_length
    
    
    # 2 判断是不是标题
    def is_title(self,text,i):
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
    
    
    # 检查长度和关键字辅助判断caption类型
        
    def check_text_caption_type(self,input_string):
        pattern = r'(?i)(图|figure|pic|image|图像|illustrate|illustration|show|demonstrate|展示|说明|表|table|photo|照片|illustrated)(?:[:：\-—\s]*\d*)?'
        if not re.search(pattern, input_string):
            return False
        return True
    
    
    # 查看文字的布局信息辅助辅助判断caption类型
    def check_absolute_pos(self,chunk):
        if (((chunk['display']['page_width']/10)*2) < chunk['display']['left']) and (chunk['display']['left']<((chunk['display']['page_width']/10)*5)):
            return True
        
        
        
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
        buffer_text = ""
        first_chunk = None
        current_page_no = None
        display_list = []  # 使用列表来存储所有相关的 display 信息
        positions_list = []  # 使用列表来存储所有相关的 position 信息

        for chunk in chunk_list:
            try:
                if chunk['type'] == 'CAPTION':
                    chunk['type'] = 'SECTION_TEXT'
                if chunk['type'] == 'SECTION-TEXT':
                    if self.calculate_length(chunk['text']) < 500:
                        # 如果长度小于500，直接添加到结果列表
                        chunk['text'] = chunk['text']+'\n'
                        result_chunks.append(chunk)
                    else:
                        # 长度超过500，进行拆分处理
                        buffer_text = chunk['text']
                        extra_info = ''
                        current_length = self.calculate_length(buffer_text)
                        while current_length > 500:
                            temp_text = ''
                            for word in re.split(r'\s+', buffer_text):
                                if self.calculate_length(extra_info + temp_text + ' ' + word) <= 500:
                                    temp_text += ' ' + word
                                else:
                                    break
                            merged_chunk = copy.deepcopy(chunk)
                            merged_chunk['text'] = extra_info + temp_text.strip()
                            result_chunks.append(merged_chunk)
                            buffer_text = buffer_text[len(temp_text.strip()):].strip()
                            current_length = self.calculate_length(extra_info + buffer_text)
                        if buffer_text:
                            # 处理剩余部分
                            merged_chunk = copy.deepcopy(chunk)
                            merged_chunk['text'] = extra_info + buffer_text
                            result_chunks.append(merged_chunk)
                else:
                    # 非“SECTION-TEXT”类型，直接添加
                    result_chunks.append(chunk)

            except Exception as e:
                print("Error processing chunk: ", e)
            
            for chunk in result_chunks:
                if isinstance(chunk['display'], dict):
                    chunk['display'] = [chunk['display']]
                if isinstance(chunk['positions'], dict):
                    chunk['positions'] = [chunk['positions']]
                if 'children' in chunk:
                    del chunk['children']

        return result_chunks
    


    # 哈希编码
    def hash_md5(self,text):
        hash_object = hashlib.md5(text.encode())
        return hash_object.hexdigest()
    
    
    # 主函数        
    def process(self):
        
        # 1. 预处理（去除header，footer），修改section-title为section-text，
        self.preprocess()
        
        # 2. 判断Title类型（aux强制把标题添加在第一个chunk，类型为title），统一在开头位置插入一个text为file_name的chunk
        aux_chunk = {}
        aux_chunk['display'] = copy.deepcopy(self.data[0]['display'])
        aux_chunk['positions'] = copy.deepcopy(self.data[0]['positions'])
        aux_chunk['page_no'] = copy.deepcopy(self.data[0]['page_no'])
        aux_chunk['text'] = self.file_name
        aux_chunk['base64'] = None
        aux_chunk['type'] = 'TITLE'
        aux_chunk['stage'] = copy.deepcopy(self.data[0]['stage'])
        self.data.insert(0, aux_chunk)
        for i, chunk in enumerate(self.data):                
            if self.is_title(chunk['text'],i) and (chunk['type']=='SECTION-TEXT'):
                chunk['type'] = 'TITLE'
                
        # 3. 根据Title类型标记索引
        index_list = []
        for i, chunk in enumerate(self.data):
            if chunk['type']=='TITLE':
                index_list.append(i)
            
        # 4. 根据索引切分子list
        chunk_lists = []
        for i in range(len(index_list)):
            if (i+1)!=len(index_list):
                chunks = self.data[index_list[i]:index_list[i+1]]
            else:
                chunks = self.data[index_list[i]:]
            chunk_lists.append(chunks)
            
        # 5. 给每个chunk添加附加信息
        for chunk_list in chunk_lists:
            for i, chunk in enumerate(chunk_list):
                chunk['attach_text']=self.file_name + '的' + chunk_list[0]['text'] + '的部分内容为' + ':' + chunk['text']
                page_no = copy.deepcopy(chunk['positions']['page_no'])
                if(len(page_no)>1): # 处理docai的错误类型
                    chunk['positions']['page_no'] = [page_no[0]]

        for chunk_list in chunk_lists:
            for i, chunk in enumerate(chunk_list):
                if i > 0 and chunk_list[i-1]['type'] == 'IMAGE' and chunk['type'] == 'SECTION-TEXT' and (self.check_absolute_pos(chunk) or self.check_text_caption_type(chunk['text'])):
                    chunk['type'] ='CAPTION'
                if i < len(chunk_list) - 1 and chunk_list[i+1]['type'] == 'IMAGE' and chunk['type'] == 'SECTION-TEXT' and (self.check_absolute_pos(chunk) or self.check_text_caption_type(chunk['text'])):
                    chunk['type'] = 'CAPTION'
        
        
        for chunk_list in chunk_lists:
            for i, chunk in enumerate(chunk_list):
                if chunk['type']=='IMAGE':
                    text_1 = ''
                    text_2 = ''
                    if i>0 and chunk_list[i-1]['type']=="CAPTION":
                        text_1 = chunk_list[i-1]['text']
                    if i < len(chunk_list) - 1 and chunk_list[i+1]['type'] == 'CAPTION':
                        text_2 = chunk_list[i+1]['text']
                    chunk['text'] = text_1+text_2
        
        for chunk_list in chunk_lists:
            for chunk in chunk_list:
                if chunk['type'] == 'IMAGE':
                    chunk['type'] = 'SECTION-TEXT'
        # 6. 判断type为caption的chunk
        # for chunk_list in chunk_lists:
        #     for i, chunk in enumerate(chunk_list):
        #         if (chunk['type']=='CAPTION'):
        #             chunk['parent_pos'] = -1
        #         if (chunk['type']=='IMAGE'):
        #             if i!=0 and i!=len(chunk_list)-1:
        #                 if (chunk_list[i+1]['type']=='SECTION-TEXT' or chunk_list[i+1]['type']=='CAPTION')and self.check_text_caption_type(chunk_list[i+1]['text']):
        #                     chunk_list[i+1]['type'] = 'CAPTION'
        #                     chunk_list[i+1]['parent_pos'] = 1
        #                     chunk_list[i+1]['state'] = 'USED'
        #                 elif (chunk_list[i-1]['type']=='SECTION-TEXT' or chunk_list[i-1]['type']=='CAPTION')and  self.check_text_caption_type(chunk_list[i-1]['text']) and ('state' in chunk_list[i-1]):
        #                     chunk_list[i-1]['type'] = 'CAPTION'
        #                     chunk_list[i-1]['parent_pos'] = -1
        
        # 6.处理长度超过512的文本
        chunk_lists = [self.process_chunks_length(chunk_list) for chunk_list in chunk_lists]
        # for chunk_list in chunk_lists:
        #     for chunk in chunk_list:
        #         chunk['texts'] = []
        
        
        
        # 7. 合并chunk
        chunk_knowledge_list = []
        for chunk_list in chunk_lists:
            chunk_knowledge_list.append(copy.deepcopy(self.process_knowledge_chunks(chunk_list)))
        
        chunk_long_context_list = []
        for chunk_list in chunk_lists:
            chunk_long_context_list.append(copy.deepcopy(self.process_long_context_chunks(chunk_list)))
            
            
        # 8. 给每个chunk添加id信息，并删除stage
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
                chunk.pop('stage')
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
                chunk.pop('stage')
                num+=1
                
                
        # 9. 给底层chunk添加parent_id
        for chunk_list in chunk_knowledge_list:
            for i, chunk in enumerate(chunk_list):
                # if len(chunk_list)>1:
                parent_id = chunk_list[0]['id']
                chunk['page_no'] = chunk['page_no']+1
                if chunk['type'] == 'TITLE':
                    chunk['parent_id'] = parent_id
                if chunk['type'] in ('SECTION-TEXT','TABLE','IMAGE'):
                    chunk['parent_id'] = parent_id
                if chunk['type']=='CAPTION':
                    chunk['parent_id'] = chunk_list[i+chunk['parent_pos']]['id']
                    del chunk['parent_pos']
                
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
                if chunk['type']=='CAPTION':
                    chunk['parent_id'] = chunk_list[i+chunk['parent_pos']]['id']
                    del chunk['parent_pos']
                
                for cm, pos in enumerate(chunk['positions']):
                    if isinstance(pos, dict):
                        
                        chunk['positions'][cm]['bbox'] = chunk['positions'][cm]['bbox'][0]
                
                
                
                if j == 0 and i==0:
                    chunk['positions'] = chunk_list[1]['positions']
                    
                    
        # 10. 返回列表类型
        final_knowledge_chunk_list = []    
        for chunk_list in chunk_knowledge_list:
            for chunk in chunk_list:
                if chunk['type']!='TITLE':
                    self.rename_key(chunk, 'base64', 'base64_list')
                    final_knowledge_chunk_list.append(chunk)
                # final_knowledge_chunk_list.append(chunk)
                
        
        final_long_context_chunk_list = []    
        for chunk_list in chunk_long_context_list:
            for chunk in chunk_list:
                final_long_context_chunk_list.append(chunk)

        return (final_knowledge_chunk_list, final_long_context_chunk_list)

if __name__ =="__main__":
    tokenizer_path = '/root/web_demo/HybirdSearch/models/models--Qwen--Qwen1.5-14B-Chat'
    file_path = '/root/web_demo/HybirdSearch/cmx_workapace/es_app_0613/unprocessed/2022 年半年度报告.pdf.json'
    file_name = '2022 年半年度报告'   # 别带.pdf
    dp = KnowledgeDocumentPreprocessor(tokenizer_path,  file_path, file_name)
    knowledge, long_content = dp.process()
    
    with open('/root/web_demo/HybirdSearch/cmx_workapace/es_app_0613/问答_processed/2022 年半年度报告.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=4)
    with open('/root/web_demo/HybirdSearch/cmx_workapace/es_app_0613/长文本_processed/2022 年半年度报告.json', 'w', encoding='utf-8') as f:
        json.dump(long_content, f, ensure_ascii=False, indent=4)