import requests
import numpy as np
from config import Config
from typing import List, Union, Dict, Any

class EmbeddingsClient:
    def __init__(self):
        self.url = Config.EMBEDDING_SERVER_URL
        self.model = Config.EMBEDDING_SERVER_PATH
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def _make_request(self, input_text: List[str]) -> Union[List[dict], dict]:
        payload = {
            'input': input_text,
            'model': self.model,
        }

        response = requests.post(self.url, json=payload, headers=self.headers)
        if response.status_code == 200:
            return response.json()['data']
        else:
            return {'error': 'Request failed', 'status_code': response.status_code}

    def encode(self, input_text: Union[List[str], str]) -> Union[List[float]]:
        data = self._make_request(input_text)
        if isinstance(data, dict) and 'error' in data:
            return data
        if isinstance(input_text, str):
            return np.array(data[0]['embedding'])       
        embeddings = np.array([item['embedding'] for item in data])
        return embeddings
    

class RerankClient:
    def __init__(self):
        self.url = Config.RERANKER_SERVER_URL
        self.model = Config.RERANKER_SERVER_PATH
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def _make_request(self, query: str, documents: List[str], return_documents: bool) -> Union[Dict[str, Any], Dict[str, Any]]:
        payload = {
            'query': query,
            'documents': documents,
            'return_documents': return_documents,
            'model': self.model
        }

        response = requests.post(self.url, json=payload, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': 'Request failed', 'status_code': response.status_code}

    def predict(self, query: str, documents: List[str], return_documents: bool = False) -> List[float]:
        documents = [doc[:1024] for doc in documents if doc]
        result = self._make_request(query, documents, return_documents)
        if isinstance(result, dict) and 'error' in result:
            return result       
        scores = [item['relevance_score'] for item in result['results']]
        return scores
    

if __name__ == '__main__':
    # client = EmbeddingsClient()
    # input_text = ["NLP is fun"]
    # embeddings = client.encode(input_text)
    # print(f"单个文本的embedding: {embeddings}")
    # print(f"embedings shape: {len(embeddings)}")
    
        #  usage
    rerank_client = RerankClient()

    # 重排
    response = rerank_client.predict(query="relevant query", documents=['This is the first chunk of text.', 'Here is another chunk of text.', 'This chunk is very relevant to the query.', 'Irrelevant chunk of text.'])
    print(f"重排结果: {response}")