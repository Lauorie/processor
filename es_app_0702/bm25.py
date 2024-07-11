import thulac
import math
from collections import Counter

# class BM25Ranker:
#     def __init__(self):
#         # 初始化THULAC，仅分词不标注词性
#         self.thu1 = thulac.thulac(seg_only=True)

#     def tokenize(self, text):
#         return self.thu1.cut(text, text=True).split()
    
#     def calculate_idf(self, documents):
#         df = {}
#         for document in documents:
#             for word in set(document):
#                 df[word] = df.get(word, 0) + 1
#         return {word: math.log((len(documents) - freq + 0.5) / (freq + 0.5) + 1) for word, freq in df.items()}
    
#     def bm25_score(self, document, query, idf, avgdl, k1=2.0, b=0.75):
#         score = 0.0
#         doc_length = len(document)
#         for word in query:
#             if word in document:
#                 freq = document[word]
#                 idf_score = idf.get(word, 0)
#                 score += idf_score * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (doc_length / avgdl)))
#         return score
    
#     def rank(self, chunks, query_text):
#         # 预处理文档和查询
#         documents = [self.tokenize(chunk['_source']['file_docai_json']['text']) for chunk in chunks]
#         query = self.tokenize(query_text)
        
#         # 计算IDF和平均文档长度
#         idf = self.calculate_idf(documents)
#         avgdl = sum(len(doc) for doc in documents) / len(documents)
        
#         # 计算每个chunk的BM25得分
#         doc_freqs = [Counter(doc) for doc in documents]
#         scored_chunks = [(self.bm25_score(doc, query, idf, avgdl), chunk) for doc, chunk in zip(doc_freqs, chunks)]
        
#         # 根据得分排序chunks
#         scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
#         # 返回排序后的chunks
#         return [chunk for _, chunk in scored_chunks]
    
class BM25Ranker_attach:
    def __init__(self):
        # 初始化THULAC，仅分词不标注词性
        self.thu1 = thulac.thulac(seg_only=True)

    def tokenize(self, text):
        return self.thu1.cut(text, text=True).split()
    
    def calculate_idf(self, documents):
        df = {}
        for document in documents:
            for word in set(document):
                df[word] = df.get(word, 0) + 1
        return {word: math.log((len(documents) - freq + 0.5) / (freq + 0.5) + 1) for word, freq in df.items()}
    
    def bm25_score(self, document, query, idf, avgdl, k1=2.0, b=0.75):
        score = 0.0
        doc_length = len(document)
        for word in query:
            if word in document:
                freq = document[word]
                idf_score = idf.get(word, 0)
                score += idf_score * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (doc_length / avgdl)))
        return score
    
    # def rank(self, query_text, chunks):
    #     # 预处理文档和查询
    #     documents = [self.tokenize(chunk['_source']['file_docai_json']['text'][:10]) for chunk in chunks]
    #     query = self.tokenize(query_text)
        
    #     # 计算IDF和平均文档长度
    #     idf = self.calculate_idf(documents)
    #     avgdl = sum(len(doc) for doc in documents) / len(documents)
        
    #     # 计算每个chunk的BM25得分
    #     doc_freqs = [Counter(doc) for doc in documents]
    #     scored_chunks = [(self.bm25_score(doc, query, idf, avgdl), chunk) for doc, chunk in zip(doc_freqs, chunks)]
        
    #     # 根据得分排序chunks
    #     scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
    #     # 返回排序后的chunks
    #     return [chunk for _, chunk in scored_chunks]
    def rank(self, query_text, chunks):
        # 预处理文档和查询
        documents = []
        for chunk in chunks:
            try:
                document = self.tokenize(chunk['_source']['file_docai_json']['attach_text'])
            except:
                document = ''
            documents.append(document)
        # documents = [self.tokenize(chunk['_source']['file_docai_json']['attach_text']) for chunk in chunks]
        query = self.tokenize(query_text)
        
        # 计算IDF和平均文档长度
        idf = self.calculate_idf(documents)
        avgdl = sum(len(doc) for doc in documents) / len(documents)
        
        # 计算每个chunk的BM25得分
        doc_freqs = [Counter(doc) for doc in documents]
        scored_chunks = [(self.bm25_score(doc, query, idf, avgdl), chunk) for doc, chunk in zip(doc_freqs, chunks)]
        
        # 根据得分排序chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # 在每个chunk中添加bm25得分属性
        sorted_chunks = []
        for score, chunk in scored_chunks:
            # 复制chunk的字典以避免修改原始数据
            modified_chunk = dict(chunk)
            modified_chunk['_source']['file_docai_json']['bm25_score'] = score
            sorted_chunks.append(modified_chunk)
        
        # 返回添加了得分的排序后的chunks
        return sorted_chunks
    
    def query_text_check(self, query, text):
        # 对 query 和 text 进行分词
        query_tokens = self.thu1.cut(query, text=False)
        text_tokens = self.thu1.cut(text, text=False)

        # 将query的分词结果转换成集合，方便查找
        query_set = set(token[0] for token in query_tokens)

        # 计算text中有多少分词未出现在query的分词结果中
        unmatched_count = sum(1 for token in text_tokens if token[0] not in query_set)

        # 计算text分词总数
        total_text_tokens = len(text_tokens)

        # 判断未匹配的分词比例是否超过40%
        if total_text_tokens > 0 and unmatched_count / total_text_tokens > 0.5:
            return True
        else:
            return False