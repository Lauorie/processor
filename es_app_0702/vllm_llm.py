import re
import json
from typing import List
from loguru import logger
from config import Config
from openai import OpenAI
from prompts import PROMPTS

logger.add(Config.MODEL_LOG_PATH)

QUERY_REWRITE_PROMPT = PROMPTS.QUERY_REWRITE_PROMPT
KNOWLEDGE_BASE_PROMPT = PROMPTS.KNOWLEDGE_BASE_PROMPT
REWRITE_QUERY_FROM_HISTORY_PROMPT = PROMPTS.REWRITE_QUERY_FROM_HISTORY_PROMPT


class LLM(object):
    def __init__(self):
        self.model_path = Config.MODEL_PATH
        self.client = OpenAI(
            api_key = Config.API_KEY,
            base_url = Config.BASE_URL,
        )        
        
    # 输入对话，输出流式结果
    def streaming_answer(self, messages):
        streamer = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=0.3,
            stream=True,
        )
        for res in streamer:
            chunk_message = res.choices[0].delta
            # 输出的第一个字符是None，所以要去掉
            if not chunk_message.content:
                continue
            yield chunk_message.content
        
    def generate_queries(self, query):
        fmt_prompt = QUERY_REWRITE_PROMPT.format(query)
        responses = self.client.chat.completions.create(
            model=self.model_path,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": fmt_prompt},
            ],
            temperature=0.3,
        )  
        # print(response.choices[0].message.content)   response = client.chat.completions.create
        query_group = ['0. ' + query] + responses.choices[0].message.content.split("\n") # ['0. query', '1. query', '2. query', '3. query']
        return query_group



    def get_answer_from_pdf(self, context:List[str], query:str):
        """Generate an answer based on the input context and query"""
        content = "\n".join(context)
        content = KNOWLEDGE_BASE_PROMPT.format(content, query)
        content = content[:25000] # 限制输入文本的长度
        messages = [{"role": "user", "content": content}]
        logger.info(f"输入模型的文本为: {messages}")
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content



    def rewrite_query_based_on_history(self, history: List[dict], query: str):
        if len(history) == 0:
            return query
        else:
            conversation = "\n".join([item['content'] for item in history])
            formatted_content = REWRITE_QUERY_FROM_HISTORY_PROMPT.format(query,conversation)
            messages = [{"role": "user", "content": formatted_content}]
            logger.info(f"输入模型的文本为: {messages}")
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content
    
    
    def _chat_stream(self, query:str, context:List[str]):
        content = "\n".join(context)
        content = KNOWLEDGE_BASE_PROMPT.format(content, query)
        content = content[:25000] # 限制输入文本的长度
        messages = [{"role": "user", "content": content}]
        logger.info(f"输入模型的文本为: {messages}")
        streamer = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=0.3,
            stream=True,
        )
        answer = ''
        for response in streamer:
            chunk_message = response.choices[0].delta
            if not chunk_message.content:
                continue
            yield chunk_message.content
            answer += chunk_message.content
        logger.info(f"====针对问题： {query}=====》的回答为: {answer}")



    # 流式输出有历史信息        
    def _chat_stream_with_history(self, query:str, context:List[str], history:List[dict]=None):
        logger.info(f"用户当前问题为: {query}")
        if history:
            logger.info(f"用户启用了历史，当前的历史信息为: {history}")
            rewritten_query = self.rewrite_query_based_on_history(history, query)           
            logger.info(f"用户问题经过历史信息重写后为: {rewritten_query}")                    
            content = "\n".join(context)
            content = self.KNOWLEDGE_BASE_PROMPT.format(content, rewritten_query)
            content = content[:25000] # 限制输入文本的长度
            messages = [{"role": "user", "content": content}]
            logger.info(f"输入模型的文本为: {messages}")
            streamer = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.3,
                stream=True,
            )
            answer = ''
            for response in streamer:
                chunk_message = response.choices[0].delta
                # 输出的第一个字符是None，所以要去掉
                if not chunk_message.content:
                    continue
                yield chunk_message.content
                answer += chunk_message.content
            logger.info(f"针对问题： {query}=====》回答为: {answer}")
        else:
            self._chat_stream(query, context)
        
    # 测试模型流式输出
    def chat_stream(self, query):
        conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'},]
        conversation.append({'role': 'user', 'content': query})
        return self.streaming_answer(conversation)

    def chat_stream_with_router(self, query, history):
        conversation = [{'role': 'system', 'content': '你是智工知语，你能理解人类语言。作为用户的工作小助手，你能解答用户的任何问题!'}]
        conversation.extend(history) if history else None       
        conversation.append({'role': 'user', 'content': query})
        logger.info(f"Rounter走模型, query+his的输入为: {conversation}")
        return self.streaming_answer(conversation)

        
    # RAG数据追加到文件
    def _append_data_to_file(self, file_path, new_data):   
        wrapped_data = {  
            "conversations": new_data  
        }  
    
        try:  
            # 尝试读取现有数据  
            with open(file_path, 'r', encoding='utf-8') as f:  
                file_content = f.read()  
                if file_content:  
                    try:  
                        # 解析文件内容为JSON  
                        existing_data = json.loads(file_content)  
                    except json.JSONDecodeError:  
                        # 如果解析失败，则初始化为空列表  
                        logger.error("在保存对话记录时 JSON 解析错误：文件内容不是一个有效的 JSON 格式")  
                        existing_data = []  
                else:  
                    # 如果文件内容为空，则初始化为空列表  
                    existing_data = []  
        except FileNotFoundError:  
            # 如果文件不存在，则初始化为空列表  
            logger.warning(f"文件 {file_path} 未找到，从空数据列表开始。")  
            existing_data = []  
    
        # 将新数据添加到现有列表中  
        existing_data.append(wrapped_data)  
    
        # 将更新后的数据写回文件  
        with open(file_path, 'w', encoding='utf-8') as f:  
            json.dump(existing_data, f, ensure_ascii=False, indent=4)  
        logger.info(f"已将更新后的RAG数据保存到 {file_path}，现有数据长度：{len(existing_data)}")
        
    def re_router(self, query):
        patterns = {
            "translation": [
                re.compile(r"^翻译我们的对话信息\s*(.*)", re.MULTILINE),
                re.compile(r"^请你翻译一下刚刚的对话\s*(.*)", re.MULTILINE),
                re.compile(r"^翻译之前的对话\s*(.*)", re.MULTILINE),
                re.compile(r"^翻译对话\s*(.*)", re.MULTILINE),
                re.compile(r"^翻译我们的对话内容\s*(.*)", re.MULTILINE),  
                re.compile(r"^将对话内容翻译成\s*(.*)", re.MULTILINE),  
                re.compile(r"^请转换我们的对话为\s*(.*)", re.MULTILINE),  
                re.compile(r"^将对话转换为\s*(.*)", re.MULTILINE),  
                re.compile(r"^转换对话内容为\s*(.*)", re.MULTILINE),  
                re.compile(r"^请求翻译最近的对话\s*(.*)", re.MULTILINE),  
                re.compile(r"^把对话翻译成\s*(.*)", re.MULTILINE),  
                re.compile(r"^请翻译对话内容\s*(.*)", re.MULTILINE),  
                re.compile(r"^用另一种语言解释对话\s*(.*)", re.MULTILINE),  
                re.compile(r"^将对话翻译成其他语言\s*(.*)", re.MULTILINE), 
                re.compile(r"^将对话翻译为\s*(.*)", re.MULTILINE),
                re.compile(r"^翻译上述内容\s*(.*)", re.MULTILINE),
                re.compile(r"^把摘要翻译成英语\s*(.*)", re.MULTILINE),
                re.compile(r"^请将对话翻译成英文\s*(.*)", re.MULTILINE),
            ],
            "summary": [
                re.compile(r"^请你总结我们的对话信息\s*(.*)", re.MULTILINE),
                re.compile(r"^总结一下刚才的对话\s*(.*)", re.MULTILINE),
                re.compile(r"^总结之前的对话\s*(.*)", re.MULTILINE),
                re.compile(r"^总结对话\s*(.*)", re.MULTILINE),
                re.compile(r"^请简述我们的对话\s*(.*)", re.MULTILINE),  
                re.compile(r"^对话内容简要说明\s*(.*)", re.MULTILINE),  
                re.compile(r"^对话的摘要\s*(.*)", re.MULTILINE),  
                re.compile(r"^请提供对话的概览\s*(.*)", re.MULTILINE),  
                re.compile(r"^对话的快速总结\s*(.*)", re.MULTILINE),  
                re.compile(r"^简要描述我们的对话\s*(.*)", re.MULTILINE),  
                re.compile(r"^对话的重点\s*(.*)", re.MULTILINE),  
                re.compile(r"^请概括对话\s*(.*)", re.MULTILINE),  
                re.compile(r"^对话的主要观点\s*(.*)", re.MULTILINE),  
                re.compile(r"^对话的精髓\s*(.*)", re.MULTILINE),  
            ],
            "table": [
                re.compile(r"^请你将我们的对话信息整理成表格\s*(.*)", re.MULTILINE),
                re.compile(r"^把刚才的对话整理成表格\s*(.*)", re.MULTILINE),
                re.compile(r"^将之前的对话信息整理成表格\s*(.*)", re.MULTILINE),
                re.compile(r"^整理对话成表格\s*(.*)", re.MULTILINE),
                re.compile(r"^对话内容整理成表格形式\s*(.*)", re.MULTILINE),  
                re.compile(r"^请将数据整理成表格\s*(.*)", re.MULTILINE),  
                re.compile(r"^对话的表格形式\s*(.*)", re.MULTILINE),  
                re.compile(r"^整理对话内容为表格\s*(.*)", re.MULTILINE),  
                re.compile(r"^对话信息的表格版\s*(.*)", re.MULTILINE),  
                re.compile(r"^将对话信息转化为表格\s*(.*)", re.MULTILINE),  
                re.compile(r"^请制作对话的表格\s*(.*)", re.MULTILINE),  
                re.compile(r"^对话的表格呈现\s*(.*)", re.MULTILINE),  
                re.compile(r"^以表格形式展现对话\s*(.*)", re.MULTILINE),  
                re.compile(r"^将对话整理为结构化表格\s*(.*)", re.MULTILINE), 
                re.compile(r"^用表格表示上述结果\s*(.*)", re.MULTILINE),
                re.compile(r"^使用表格整理上述结果\s*(.*)", re.MULTILINE),
                re.compile(r"^请将对话整理成表格形式\s*(.*)", re.MULTILINE),
                re.compile(r"^请用表格的形式列举出来\s*(.*)", re.MULTILINE),
            ],
        }

        for action, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = pattern.search(query)
                if match:
                    return action
        return "rag"
    
    def build_request_message(self, action, history):  
        history = '\n'.join([f"{i['role']}: {i['content']}" for i in history]) if history else ""  
        content = f"{action}下面的对话信息:\n{history}"  
        messages = [{"role": "user", "content": content}]
        return messages, content

    def process_request(self, action, messages):
        return self.streaming_answer(messages)
    
    def get_translation_answer(self, history):  
        messages, content = self.build_request_message("英汉互译：", history)  
        return self.process_request("翻译", messages)  
  
    def get_summary_answer(self, history):  
        messages, content = self.build_request_message("总结", history)  
        return self.process_request("总结", messages)  
  
    def get_table_answer(self, history):  
        messages, content = self.build_request_message("整理成表格", history)  
        return self.process_request("整理成表格", messages)


if __name__ == "__main__":
    llm = LLM()
    query = "交叉二楼在哪？"
    # history = [{"role":"user","content":'什么是自然语言处理？'},
    # {"role":"assistant","content":'自然语言处理是一门研究人类语言的学科。'},
    # {"role":"user","content":'自然语言处理有哪些应用？'},
    # {"role":"assistant","content":'自然语言处理有很多应用，比如机器翻译、语音识别等。'}]
    # context = ['人工智能助手','交叉二楼位于复旦大学张江校区，是一个学术交流的场所。','交叉二楼的开放时间是周一到周五，上午9点到下午5点。']
    
    result = llm.generate_queries(query)
    # for res in result:
    #     print(res)
    print(result)
    
