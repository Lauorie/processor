import re
import torch
import json
from rouge import Rouge
from loguru import logger
from flask import Response
from config import Config
from embeddings import RerankClient
from typing import List, Tuple, Dict



class Reranker:
    def __init__(self):
        self.model = RerankClient()

    def rerank(self, query: str, docs: List[List[str]], k=10, threshold=0.3) -> List[str]:
        """Rerank a list of documents based on a query using a reranking model.
        Args:
            docs: The list of documents to rerank.
        Returns:
            The reranked list of documents.   
        """
        docs_ = [item for item in docs[0]]
        docs = list(set(docs_))
        scores = self.model.predict(query, docs)

        docs_scores = [(docs[i], scores[i]) for i in range(len(docs))]
        
        # Filter out documents with scores below the threshold
        filtered_docs_scores = [doc_score for doc_score in docs_scores if doc_score[1] >= threshold]
        
        # If no documents exceed the threshold, return an empty list
        if not filtered_docs_scores:
            return []
        
        # Sort the filtered documents by score in descending order
        filtered_docs_scores = sorted(filtered_docs_scores, key=lambda x: x[1], reverse=True)
        
        # Return the top k documents
        return [item[0] for item in filtered_docs_scores[:k]]
        
    def rerank_docai_chunk(self, json_data: List[Dict[str, str]], response: str, threshold=0.3) -> Response:
        for chunk_data in json_data:
            if 'text_list' in chunk_data and 'positions' in chunk_data and 'base64_list' in chunk_data:
                text_list = chunk_data['text_list']
                positions = chunk_data['positions']
                base64_list = chunk_data['base64_list']

                scores = self.model.predict(response, text_list)

                # Combine the lists with their scores
                combined = list(zip(text_list, positions, base64_list, scores))
                # [(text, position, base64, score), ...]
                
                # Filter based on the threshold
                filtered_combined = [item for item in combined if item[3] >= threshold]

                if not filtered_combined:
                    # If no sentences exceed the threshold, use top 3 of the original combined list
                    sorted_combined = sorted(combined, key=lambda x: x[3], reverse=True)[:3]
                else:
                    # Sort the filtered sentences by score in descending order
                    sorted_combined = sorted(filtered_combined, key=lambda x: x[3], reverse=True)

                # Update the chunk_data with the sorted data
                chunk_data['text_list'] = [item[0] for item in sorted_combined]
                chunk_data['positions'] = [item[1] for item in sorted_combined]
                chunk_data['base64_list'] = [item[2] for item in sorted_combined]
        
        logger.info(f"JSON_DATA 已经过RERANKER重排, 排序后的长度: {len(json_data)}")
        # Return json data in a Response object
        return Response(json.dumps(json_data, ensure_ascii=False, indent=4), mimetype='application/json')
    
    def rerank_processor_chunk(self, json_data: List[Dict[str, str]], response: str, threshold=0.7) -> Response:
        chunk_texts = [item['chunk_text'] for item in json_data]
        scores = self.model.predict(response, chunk_texts)
        # Combine the chunk texts with their scores
        combined = list(zip(json_data, scores))

        # Filter based on the threshold
        filtered_combined = [item for item in combined if item[1] >= threshold]

        if not filtered_combined:
            # If no chunks exceed the threshold, use the original combined list
            sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        else:
            # Sort the filtered chunks by score in descending order
            sorted_combined = sorted(filtered_combined, key=lambda x: x[1], reverse=True)

        # Extract the sorted json_data
        sorted_json_data = [item[0] for item in sorted_combined]

        logger.info(f"JSON_DATA 已经过RERANKER重排, 排序后的长度: {len(sorted_json_data)}")
        # Return sorted json data in a Response object
        return Response(json.dumps(sorted_json_data, ensure_ascii=False, indent=4), mimetype='application/json')
    
    def rerank_combined_chunk(self, json_data: List[Dict[str, str]], response: str, processor_threshold=0.7, docai_threshold=0.7) -> Response:
        # 先执行 rerank_processor_chunk 的逻辑
        chunk_texts = [item['chunk_text'] for item in json_data]
        scores = self.model.predict(response, chunk_texts)
        combined = list(zip(json_data, scores))
        filtered_combined = [item for item in combined if item[1] >= processor_threshold]

        if not filtered_combined:
            sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        else:
            sorted_combined = sorted(filtered_combined, key=lambda x: x[1], reverse=True)

        sorted_json_data = [item[0] for item in sorted_combined]

        logger.info(f"JSON_DATA 已经过 rerank_processor_chunk 重排, 排序后的长度: {len(sorted_json_data)}")

        # 使用重排后的数据进行 rerank_docai_chunk 的逻辑
        for chunk_data in sorted_json_data:
            if 'text_list' in chunk_data and 'positions' in chunk_data and 'base64_list' in chunk_data:
                text_list = chunk_data['text_list']
                positions = chunk_data['positions']
                base64_list = chunk_data['base64_list']

                scores = self.model.predict(response, text_list)
                combined = list(zip(text_list, positions, base64_list, scores))
                filtered_combined = [item for item in combined if item[3] >= docai_threshold]

                if not filtered_combined:
                    sorted_combined = sorted(combined, key=lambda x: x[3], reverse=True)[:3]
                else:
                    sorted_combined = sorted(filtered_combined, key=lambda x: x[3], reverse=True)

                chunk_data['text_list'] = [item[0] for item in sorted_combined]
                chunk_data['positions'] = [item[1] for item in sorted_combined]
                chunk_data['base64_list'] = [item[2] for item in sorted_combined]

        logger.info(f"JSON_DATA 已经过 rerank_docai_chunk 重排, 去除了不相关的图片！")

        # 返回重排后的 json 数据
        return Response(json.dumps(sorted_json_data, ensure_ascii=False, indent=4), mimetype='application/json')

if __name__ == "__main__":
    reranker = Reranker()
    
    # 测试数据
    json_path = "/root/web_demo/HybirdSearch/es_app_0624/小米SU7用户手册_reranked.pdf.json"
    
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        test_json_data = json.load(f)

    test_response = """
    更换小米SU7的遥控钥匙电池时，您需要按照以下步骤操作:
    1.使用塑料工具撬开遥控钥匙顶部饰条并去除左右侧外壳。
    2.手动打开遥控钥匙电池保护盖，拆除旧遥控钥匙电池。
    3.安装新的CR2450型号的遥控钥匙电池，确保电池正极朝上。
    4.重新盖上电池保护盖，
    5.安装遥控钥匙左右侧外壳和顶部饰条。在更换过程中请注意消除静电，防止损坏电路板，并确保使用原装电池证性能。如果电池电量低于30%，中控屏会提示更换。
    """
    # 重新排序 JSON 数据
    reranked_response = reranker.rerank_docai_chunk(test_json_data, test_response, threshold=0.7)
    reranked_response = json.loads(reranked_response.get_data(as_text=True))
    save_path = "/root/web_demo/HybirdSearch/es_app_0624/小米SU7用户手册_processed_reranked.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(reranked_response, f, ensure_ascii=False, indent=4)
        print("JSON 数据已经过 RERANKER 重排。")
    # 打印重新排序后的 JSON 数据
    # print(reranked_response.get_data(as_text=True))

class RougeReranker:
    def __init__(self, rouge_type:str='rouge-2', rouge_score_threshold:float=0.1):
        self.rouge = Rouge()
        self.rouge_type = rouge_type
        self.rouge_score_threshold = rouge_score_threshold

    def calculate_rouge(self, query: str, chunk: str) -> float:
        """
        计算查询和文本块之间的 ROUGE 分数。
        
        参数:
            query (str): 原始查询。
            chunk (str): 搜索结果中的文本块。
            
        返回:
            float: ROUGE-L 的 fmeasure 分数。
        """
        if not isinstance(query, str) or not isinstance(chunk, str):
            raise ValueError("查询和文本块都必须是字符串")
        
        query = " ".join(list(query))
        chunk = " ".join(list(chunk))
        
        scores = self.rouge.get_scores(query, chunk)[0][self.rouge_type]['f']
        
        return scores

    def rank_chunks(self, query: str, chunks: List[str], threshold: float = 0.1) -> List[Tuple[str, float]]:
        """
        根据 ROUGE 分数对文本块进行排序并过滤低于阈值的文本块。
        
        参数:
            query (str): 原始查询。
            chunks (List[str]): 文本块列表。
            threshold (float): ROUGE-L 的 fmeasure 分数阈值，低于此值的文本块会被过滤掉。
            
        返回:
            List[Tuple[str, float]]: 包含文本块及其 ROUGE 分数的元组列表。
        """
        scores = [(chunk, self.calculate_rouge(query, chunk)) for chunk in chunks]
        # 按照 ROUGE-L 分数降序排序
        ranked_chunks = sorted(scores, key=lambda x: x[1], reverse=True)
        # 过滤低于阈值的文本块
        filtered_chunks = [(chunk, score) for chunk, score in ranked_chunks if score >= threshold]
        return filtered_chunks

    def detect_language(self, text: str) -> str:
        """
        根据文本中字符的比例判断语言是中文还是英文。
        
        参数:
            text (str): 要判断的字符串。
            
        返回:
            str: 'chinese' 或 'english'。
        """
        if not text:
            return "unknown"
        
        if not isinstance(text, str):
            raise ValueError("Expected a string or bytes-like object")
        cleaned_text = re.sub(r'[\d!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text) # 替换掉标点符号

        # 计算英文长度
        english_words = re.findall(r'[a-zA-Z]+', cleaned_text)
        english_word_count = len(english_words)

        # 计算中文长度
        chinese_characters = re.findall(r'[\u4e00-\u9fff]', cleaned_text)
        chinese_character_count = len(chinese_characters)
        
        # 如果中文字符数大于等于英文单词数，判断为中文
        if chinese_character_count >= english_word_count:
            return "chinese"
        else:
            return "english"

    # 服务中的示例用法：
    def rerank_jsondata_old(self, json_data: List[Dict[str, str]], response: str) -> Response :
        response_language = self.detect_language(str(response))
        
        if response_language not in {"chinese", "english"}:
            # 如果 response 语言不是中文或英文，不进行重排
            logger.info(f"response 语言不是中文或英文，不进行ROUGE重排")
            return Response(json.dumps(json_data, ensure_ascii=False, indent=4), mimetype='application/json')
        
        for chunk in json_data:
            chunk_text = str(chunk['chunk_text'])
            chunk_language = self.detect_language(chunk_text)
            # 如果 chunk 语言不是中文或英文，不进行重排       
            if chunk_language not in {"chinese", "english"}:
                logger.info(f"chunk 语言不是中文或英文，不进行ROUGE重排")
                return Response(json.dumps(json_data, ensure_ascii=False, indent=4), mimetype='application/json')
            
            # 如果 response 和 chunk 语言不一致，不进行重排
            if response_language != chunk_language:
                logger.info(f"response 和 chunk 语言不一致，不进行ROUGE重排")
                return Response(json.dumps(json_data, ensure_ascii=False, indent=4), mimetype='application/json')
        
        # 如果所有的 chunks 和 response 语言一致，进行排序
        chunks = [str(chunk['chunk_text']) for chunk in json_data]
        ranked_chunks = self.rank_chunks(response, chunks)
        sorted_json_data = [chunk for chunk_text, _ in ranked_chunks for chunk in json_data if chunk['chunk_text'] == chunk_text]
        logger.info(f"JSON_DATA 已经过ROUGE重排, 排序后的长度: {len(sorted_json_data)}")
        sorted_json_data = json.dumps(sorted_json_data, ensure_ascii=False, indent=4)

        return Response(sorted_json_data, mimetype='application/json')

    def rerank_jsondata(self, json_data: List[Dict[str, str]], response: str) -> Response :
        """
        Re-rank the json data based on ROUGE scores with respect to the user's query,
        only for sentences where the language matches the query's language.
        """
        response_language = self.detect_language(str(response))
        
        if response_language not in {"chinese", "english"}:
            # 如果 response 语言不是中文或英文，不进行重排
            logger.info(f"response 语言不是中文或英文，不进行ROUGE重排")
            return Response(json.dumps(json_data, ensure_ascii=False, indent=4), mimetype='application/json')
        
        """
        json_data = [{"chunk_text": "文本块1", 
                    "text_list": ["文本块1的句子1", "文本块1的句子2", ...]
                    "positions": [0, 1, ...],
                    "base64_list": ["base64编码的图片1", "base64编码的图片2", ...]},] 
        """
        for chunk_data in json_data:
            if chunk_data.get('text_list') and chunk_data.get('positions') and chunk_data.get('base64_list'):
                text_list = chunk_data['text_list']
                positions = chunk_data['positions']
                base64_list = chunk_data['base64_list']

                new_text_list, new_positions, new_base64_list = [], [], []

                # Process each sentence in the chunk
                for sentence, position, base64 in zip(text_list, positions, base64_list):
                    sentence_language = self.detect_language(sentence)

                    # Check if the language of the sentence matches the response
                    if sentence_language == response_language:
                        new_text_list.append((sentence, position, base64, self.calculate_rouge(response, sentence)))
                    else:
                        new_text_list.append((sentence, position, base64, 0))  # Keep original order for non-matching languages

                # Sort by ROUGE score where languages matched, keep original order otherwise
                sorted_sentences = sorted(new_text_list, key=lambda x: x[3], reverse=True)

                # Update the chunk_data with the sorted data
                chunk_data['text_list'] = [item[0] for item in sorted_sentences]
                chunk_data['positions'] = [item[1] for item in sorted_sentences]
                chunk_data['base64_list'] = [item[2] for item in sorted_sentences]
        logger.info(f"JSON_DATA 已经过ROUGE重排, 排序后的长度: {len(json_data)}")
        # Return json data in a Response object
        return Response(json.dumps(json_data, ensure_ascii=False, indent=4), mimetype='application/json')
        # return json_data







    



# if __name__ == "__main__":
#     RougeReranker = RougeReranker()
    
#     json_path = "/root/web_demo/HybirdSearch/z和利时文档/2022 年半年度报告_processed.json"
#     with open(json_path, 'r', encoding='utf-8') as f:
#         json_data = json.load(f)
#     response = "朗新公司服务能源领域近25年。"
    
#     reranked_json_data = RougeReranker.rerank_jsondata(json_data, response)
#     with open("/root/web_demo/HybirdSearch/z和利时文档/2022 年半年度报告_processed_reranked.json", 'w', encoding='utf-8') as f:
#         json.dump(reranked_json_data, f, ensure_ascii=False, indent=4)
#         print("JSON 数据已经过 ROUGE 重排。")











# # for reranker testing
# if __name__ == "__main__":
#     model_path = "/root/web_demo/HybirdSearch/models/models--BAAI--bge-reranker-large"
#     reranker = Reranker(model_path)
    
#     query = "上海有哪些著名的旅游景点?"

#     docs = [
#         [
#             "北京是中国的首都,拥有众多著名的旅游景点,如故宫、长城、天安门广场等。故宫是明清两代的皇宫,也是世界上最大的宫殿建筑群。",
#             "上海是中国最大的城市,以其现代化的都市风貌和丰富的旅游资源而闻名。著名的景点包括外滩、东方明珠塔、豫园等。",
#             "西安是中国著名的古都,拥有悠久的历史和丰富的文化遗产。这里的主要旅游景点包括兵马俑、古城墙、大雁塔等。",
#             "杭州是一座美丽的城市,以西湖著称。西湖是中国最著名的景点之一,以其秀丽的湖光山色和众多的历史遗迹而闻名。",
#             "桂林是中国著名的旅游城市,以其独特的喀斯特地貌和秀丽的山水景观而闻名。著名的景点包括漓江、阳朔西街、象鼻山等。",
#             "成都是四川省的省会,以其悠闲的生活方式和美味的食物而闻名。这里的主要旅游景点包括武侯祠、锦里、宽窄巷子等。"
#         ]
#     ]
    
#     reranked_docs = reranker.rerank(query, docs, 3)
#     for i, doc in enumerate(reranked_docs):
#         print(f"{i+1}. {doc}")