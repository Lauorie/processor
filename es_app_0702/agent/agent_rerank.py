from config import Config
from sentence_transformers import CrossEncoder
from loguru import logger
from embeddings import RerankClient

class AgentReranker:
    def __init__(self):
        self.model = RerankClient()

    # def rerank(self, query: str, chunks, k=10, threshold=0.000000000000000000001):
    #     """Rerank a list of documents based on a query using a reranking model.
    #     Args:
    #         chunks: The list of document chunks to rerank. Each chunk is a dictionary.
    #     Returns:
    #         The reranked list of chunks.
    #     """
    #     # Remove duplicates based on document text
    #     seen_texts = set()
    #     unique_chunks = []
    #     for chunk in chunks:
    #         doc_text = chunk['_source']['file_docai_json']['attach_text']
    #         if doc_text not in seen_texts:
    #             seen_texts.add(doc_text)
    #             unique_chunks.append(chunk)
        
    #     # Create pairs of query and document text for the model
    #     pairs = [[query, chunk['_source']['file_docai_json']['attach_text']] for chunk in unique_chunks]
        
    #     # Use the CrossEncoder model to predict scores
    #     scores = self.model.predict(pairs)

    #     # Associate each unique chunk with its score
    #     chunk_scores = [(unique_chunks[i], scores[i]) for i in range(len(unique_chunks))]
        
    #     # Filter out chunks with scores below the threshold
    #     filtered_chunk_scores = [chunk_score for chunk_score in chunk_scores if chunk_score[1] >= threshold]
        
    #     # If no chunks exceed the threshold, return an empty list
    #     if not filtered_chunk_scores:
    #         return []
        
    #     # Sort the filtered chunks by score in descending order
    #     sorted_chunk_scores = sorted(filtered_chunk_scores, key=lambda x: x[1], reverse=True)
        
    #     # Return the top k chunks
    #     return [item[0] for item in sorted_chunk_scores[:k]]
    
    def rerank(self, query: str, chunks, k=10, threshold=1e-21):

        # Remove duplicates based on document text
        seen_texts = set()
        unique_chunks = []
        for chunk in chunks:
            doc_text = chunk['_source']['file_docai_json']['attach_text']
            if doc_text not in seen_texts:
                seen_texts.add(doc_text)
                unique_chunks.append(chunk)

        # pairs = [[query, chunk['_source']['file_docai_json']['attach_text']] ]
        texts = [chunk['_source']['file_docai_json']['attach_text']for chunk in unique_chunks] 

        scores = self.model.predict(query,texts)
        
        # Associate each unique chunk with its score
        filtered_chunk_scores = []
        for i in range(len(unique_chunks)):
            unique_chunks[i]['_source']['file_docai_json']['re_score'] = scores[i]
            filtered_chunk_scores.append(unique_chunks[i])

        
        # Return the top k chunks
        return filtered_chunk_scores
        