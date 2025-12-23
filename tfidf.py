from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data import FAQs

def get_questions():
    return [i['question'] for i in FAQs]

def get_tfidf_matrix():
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(get_questions())
    return tfidf, matrix
    
def transform_query(tfidf, query):
    return tfidf.transform([query])

def get_cosine_similarity(tfidf, matrix, query):
    transformed_query = transform_query(tfidf, query)
    return cosine_similarity(transformed_query, matrix).flatten()

def get_3_best_answers(tfidf, matrix, query):
    result = []
    similarities = get_cosine_similarity(tfidf, matrix, query)
    top_indices = np.argsort(-similarities)[:3]
    
    for index in top_indices:
        question = FAQs[index]['question']
        answer = FAQs[index]['answer']
        score = similarities[index]
        result.append({
            "question": question,
            "answer": answer,
            "score": score
        })
        
    return result
    
if __name__ == "__main__":
    pass