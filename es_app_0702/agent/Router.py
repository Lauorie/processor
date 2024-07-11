import re
from Translate_agent import Translate_Agetnt
from Search_agent import Search_Agent
from Summary_agent import Summary_Agetnt
from Backup_agent import Backup_Agetnt
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from Text_split import TextSplitter
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from threading import Thread
import json
from loguru import logger
from flask import Flask, Response, request, jsonify, stream_with_context
from vllm_llm import LLM
from esearch import Indexer
import itertools
from config import Config
from itertools import chain
import requests
import json


from agent_re_patterns import translate_patterns, judge_history_relevance_patterns, retrieval_content_patterns, retrieval_position_patterns, page_patterns, title_patterns, chunk_patterns, file_patterns, several_type, query_not_contain_page_pattern, summary_type, direct_patterns



class Router_agent:
    def __init__(self, config):
        self.config = config
        self.llm_predictor = LLM()
        self.indexer = Indexer()  

    
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

    # 1.第一层判断
    # 1.1 第一层判断翻译
    def is_translate(self,query):
        for pattern in translate_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
            
    # 1.2 第一层判断搜索
    def is_search(self,query):
        for pattern in retrieval_position_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        for pattern in retrieval_content_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    # 1.3 第一层判断摘要
    def is_summary(self, query):
        for pattern in summary_type:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    
    def is_search_content(self,query):
        for pattern in retrieval_content_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def is_search_several(self,query):
        for pattern in several_type:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    
    
    
    # 1.4 合并1.1和1.2
    def classify_query(self, query):
        if re.search(r'并翻译', query, re.IGNORECASE):
            return '翻译'
        elif self.is_search(query):
            return '搜索'
        elif self.is_summary(query):
            return '摘要'
        elif self.is_translate(query):
            return '翻译'
        else:
            return '备用agent'
        
    # 2.第二层（翻译部分）
    # 2.0 判断响应query是否需要用到历史信息
    def history_relevance(self, query):
        for pattern in judge_history_relevance_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False   
    # 2.1 两种情况(直接包含页码|间接包含页码)的找回页码
    def is_translate_page(self, query):
        for pattern in page_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
            
    # 2.2 找回整个章节
    def is_translate_title(self, query):
        for pattern in title_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
            
    # 2.3 找回单个chunk
    def is_search_chunk(self, query):
        for pattern in chunk_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    # 2.4 找回全文
    def is_translate_full(self, query):
        for pattern in file_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    # 2.5 直接处理
    def is_direct(self, query):
        for pattern in direct_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    # 2.6 合并判断
    def classify_translate_summary(self, query):
        if self.is_translate_page(query):
            return 'page'
        elif self.is_direct(query):
            return 'other'
        elif self.is_search_chunk(query):
            return 'chunk'
        elif self.is_translate_full(query):
            return 'full'
        elif self.history_relevance(query):
            return 'history'
        else:
            return 'title'
    
    # 3.第二层（检索部分）
    
    
    # 3.2 判断搜索类型
    def classify_search(self, query):
        if self.history_relevance(query):
            return 'history'
        elif self.is_search_content(query):
            return 'content'
        elif self.is_search_several(query):
            return 'several_position'
        else:
            return 'single'
    
    # 4.辅助函数
    # 4.1 判断是否包含页码信息
    def judge_query_contain_page(self,query):
        for pattern in query_not_contain_page_pattern:
            if re.search(pattern, query, re.IGNORECASE):
                return False
            else:
                return True
    
    # 4.2 辅助从query中提取页码信息
    def chinese_to_arabic(self, chinese_number):
        chinese_arabic_mapping = {
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            '十': 10, '百': 100, '千': 1000, '万': 10000
        }
        total = 0
        current = 0
        unit = 1  # 当前处理的中文数字的单位（十、百、千、万）
        num = 0   # 当前处理的数字

        # 从右到左遍历中文数字字符串
        for char in reversed(chinese_number):
            if char in chinese_arabic_mapping:
                value = chinese_arabic_mapping[char]
                if value >= 10:
                    if value > unit:
                        unit = value
                        if num == 0:
                            num = 1
                    else:
                        unit *= value
                else:
                    num = value
                    total += num * unit
                    num = 0  # 重置当前数字

        return total
    
    # 4.3 从query中提取页码信息
    def extract_page_number(self, text):
        try:
            pattern = r"(第([1234567890]+)页|第([一二三四五六七八九十百千万]+)页)"
            match = re.search(pattern, text)
            if match:
                page_number_str = match.group(2) or match.group(3)
                if page_number_str.isdigit():
                    return int(page_number_str)
                else:
                    return self.chinese_to_arabic(page_number_str)
        except:
            return None
    
    
    
        
    # 5.1获取agent处理后的prompt
    def get_agent_prompt(self, user_id, index_type, router_result, query, history_list, history_relevance, file_names):

        if history_relevance:
            last_answer = history_list[-1]['content']
        if router_result == '翻译':
            translator = Translate_Agetnt(self.indexer)
            translate_type = self.classify_translate_summary(query)

            # if translate_type in ['page','title','full']:
            #     statement = ''
            #     prompt = f"请帮我从以下用户查询中提取出核心的查询信息，并忽略所有的指令性内容。例如，对于用户查询：“翻译复旦大学建校119周年为英文”，正确的查询信息提取结果应该是“复旦大学建校119周年”。\n用户查询: \"{query}\"\n请直接输出查询信息："
            #     statements = self.llm_predictor.chat_stream(prompt)
            #     for s in statements:
            #         statement += s
            #     logger.info(f"-----翻译agent提取后的statement为：{statement}")
            
            if translate_type in ['page', 'title', 'full', 'chunk']:
                statement = ''
                statement = query
                logger.info(f"-----翻译agent提取后的statement为：{statement}")
            
            if translate_type == 'history':
                statement = last_answer
                logger.info(f"-----翻译agent提取后的statement为：{statement}")
                
            # (1).翻译页
            if translate_type == 'page':
                if self.judge_query_contain_page(query):
                    page_no = self.extract_page_number(query)
                    logger.info(f"-----有页码进行翻译")
                    logger.info(f"-----提取到的页码为{page_no}")
                    return translator.return_page(user_id, index_type, query, statement, page_no, file_names)
                else:
                    logger.info(f"-----无页码进行翻译")
                    return translator.return_page(user_id, index_type, query, statement, file_names=file_names)

            # (2).翻译XXXX所属的这一章节
            elif translate_type == 'title':
                return translator.return_title_content(user_id, index_type, query, statement,file_names)
            
            # (4).翻译XXXXX这部分内容
            elif translate_type == 'chunk':
                return translator.return_chunk(user_id, index_type, query, statement,file_names)
            
            # (5).翻译全文
            elif translate_type == 'full':
                return translator.return_full(user_id, index_type, query, statement,file_names)
            
            # (6).翻译历史
            elif translate_type == 'history':
                return ['翻译历史',f"直接翻译下面的内容:\n{statement}\n你输出的翻译结果为：",statement]
            
            # (7).直接翻译输入XXXX
            else:
                return ['直接翻译',f"直接翻译下面的内容:\n{query}\n你输出的翻译结果为：",query]
        
        
            
        # 搜索
        elif router_result == '搜索':
            searcher = Search_Agent(self.indexer)
            search_type = self.classify_search(query)
            statement = ''
            
            if self.classify_search(query) == 'history':
                statement = last_answer
            else:
                prompt = f"从给定的检索指令中提取核心内容部分。例如，对于指令：“都哪篇文章提到了复旦大学建校119周年，具体是在哪一页”，提取后的内容部分是“复旦大学建校119周年”。注意，你的任务仅是提取内容，不包括执行搜索或添加任何额外信息，如果无法从指令当中继续提取请直接返回原本的指令。请确保输出内容精确无误。\n指令：{query}\n输出关键内容为："           
                statements = self.llm_predictor.chat_stream(prompt)
                for s in statements:
                    statement +=s
                logger.info(f"提取到的statement为{statement}")
            
            # if search_type in ['several_position','history']:    # 都哪些文章提到了XXXXX
                # logger.info(f"按多篇文章的位置进行搜索")
            return searcher.return_several_file_position_information(user_id, index_type, query, statement,file_names)
            # else:   # 哪个文章提到了
            #     logger.info(f"按单篇文章的位置进行搜索")
            #     return searcher.return_one_file_position_information(user_id, index_type, query, statement,file_names)
        
        
        
        elif router_result == '摘要':
            summary = Summary_Agetnt(self.indexer)
            summary_type = self.classify_translate_summary(query)
            logger.info(f"-----摘要任务的类型是：{summary_type}")
            statement = ''
            if summary_type in ['page','title','chunk','full']:
                # 获取query的statement
                # prompt = f"提取出下面总结指令中的内容部分，并保持内容字段尽量不被修改。例如：“总结复旦大学建校119周年这个文章的第5页”提取后的内容部分是“复旦大学建校119周年这个文章的第5页”。\n指令：{query}\n注意：\n你的任务是提取总结任务的内容部分，不是进行总结，也不要扩展额外的部分。\n你输出的关键内容为："
                # statements = self.llm_predictor.chat_stream(prompt)
                # for s in statements:
                #     statement += s
                statement = query
                logger.info(f"-----总结agent提取后的statement为：{statement}")
                

            if summary_type == 'history':
                statement = last_answer
                logger.info(f"-----总结agent提取后的statement为：{statement}")
        
        
            # (1).总结XXXX所在的这一页
            # (2).总结a文件的的第XX（可阿拉伯数字，汉字不能过千）页
            if summary_type == 'page':
                if self.judge_query_contain_page(query):
                    page_no = self.extract_page_number(query)
                    logger.info(f"-----有页码进行摘要")
                    return summary.return_page(user_id, index_type, query, statement, page_no,file_names)
                else:
                    logger.info(f"-----无页码进行摘要")
                    return summary.return_page(user_id, index_type, query, statement,file_names)

            # (3).总结XXXX所属的这一章节
            elif summary_type == 'title':
                return summary.return_title_content(user_id, index_type, query, statement,file_names)
            
            # (4).总结XXXXX这部分内容
            elif summary_type == 'chunk':
                return summary.return_chunk(user_id, index_type, query, statement,file_names)
            
            # (5).翻译全文
            elif summary_type == 'full':
                return summary.return_full(user_id, index_type, query, statement,file_names)
            
            # (6).总结历史
            elif summary_type == 'history':
                return ["直接总结",f"总结下面这段文本：\n{statement}\n你输出的总结为："]
            
            # (7).直接总结输入XXXX
            else:
                return ["直接总结",f"总结下面这段文本：\n{query}\n你输出的总结为："]
    
    
    
    
    
    
    # 面向用户的
    def long_context_process(self, user_id, index_type, query, history_list=None, search_more=0, file_names=None):
        # 1.判断历史相关性这里只考虑了agent
        logger.info(f"-----用户输入的问题为：{query}")
        logger.info("-----进入agent开始判断回答这个query是否需要用到历史信息")
        history_relevance = self.history_relevance(query)
        if history_relevance:
            logger.info(f"-----长文档部分：回答这个query需要用到历史信息")
        else:
            logger.info(f"-----长文档部分：回答这个query不需要用到历史信息")
        
        # 2.根据query判断走哪个agent
        router_result = self.classify_query(query)
        logger.info(f'-----长文本的路由结果router_result为：{router_result} agent')
        
        # 3.只根据query判断走哪个agent
        if router_result != "备用agent":    
            
            # 4.获取response列表
            response = self.get_agent_prompt(user_id, index_type,router_result, query, history_list, history_relevance, file_names)
        
            if router_result == "翻译":
                logger.info(f'-----具体的任务类型是：{response[0]}')          
                if self.classify_translate_summary(query) in ['page','title','full','chunk']:
                    for text in response[2]:
                        yield '原文：'
                        yield text
                        yield "\n"
                        yield '翻译：'
                        res = self.llm_predictor.chat_stream(f"直接翻译下面的文本{text}，不要做额外的分析，你翻译的输出为：")
                        for trans in res:
                            yield trans
                        yield "\n\n"
                
                else:  # 直接翻译和翻译历史
                    logger.info(f'-----任务类型为：{self.classify_translate_summary(query)}')
                    res = self.llm_predictor.chat_stream(response[1])
                    for trans in res:
                        yield trans
                
                
                    
                    
                    
            elif router_result == "搜索":
                logger.info('4')
                res = self.llm_predictor.chat_stream(response[1])
                for trans in res:
                    yield trans
        
        
        
        
            
            elif router_result == "摘要":          
                    
                if self.classify_translate_summary(query) in ['full','title','page','chunk']:
                    logger.info(f'-----classify_translate_summary(query)的结果为：{self.classify_translate_summary(query)}')

                    # prompt = f"你是一个总结专家，需要被总结的文本是：\n{response[1]}\n你输出的总结为："
                    # logger.info(f"需要被总结的文本为{response[1]}")
                    res = self.llm_predictor.chat_stream(f"用户的问题是:{query}：\n请根据相关信息：“{response[1]}”回答用户的问题，你输出的为：")
                    for summary in res:
                        yield summary
                    
                
                else:  # 直接总结和总结历史
                    logger.info(f'-----classify_translate_summary(query)的结果为：{self.classify_translate_summary(query)}')
                    res = self.llm_predictor.chat_stream(response[1])
                    for summary in res:
                        yield summary
                
                
        
        # 备用部分
        else:
            try:
                backup_agent = Backup_Agetnt(self.indexer)
                response = backup_agent.process(user_id, index_type, query, file_names)
                res = self.llm_predictor.chat_stream(f"用户的问题是:{query}：\n请根据相关信息：“{response[1]}”回答用户的问题，你输出的为：")
                for ans in res:
                    yield ans
            # logger.info(f"调用备用agent接口")
            # # def call_streaming_api(url, payload):
            # headers = {'Content-Type': 'application/json'}
            # url = "http://localhost:5066/stream/V3/"
            # payload = {
            #     'query': query,
            #     'user_id': user_id,
            #     'index_type': index_type,
            #     'search_more': search_more,
            #     'history': history_list,
            #     'file_names': file_names
            #     }
            # try:
            #     with requests.post(url, data=json.dumps(payload), headers=headers, stream=True) as response:
            #         if response.status_code == 200:
            #             # Process each line from the response
            #             for i in response:
            #                 yield i
            #             # for line in response.iter_lines():
            #             #     if line:  # filter out keep-alive new lines
            #             #         decoded_line = line.decode('utf-8')
            #             #         yield decoded_line # Handle or process the data here
            #         else:
            #             print(f"Error: Received status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
    
        