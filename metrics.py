import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the model and tokenizer
from text_processing_utils import get_embeddings_batch, clean_text


def calculate_answer_coherence_score(answer, k, tokenizer, embedding_model):
    words = answer.split()
    similarities = []
    num_words = len(words)

    for i in range(num_words - k):
        curr_word = words[i]
        next_k_words = words[i + 1:i + k + 1]

        # Get embeddings for the current word and the next k words
        curr_word_embedding = get_embeddings_batch([curr_word], tokenizer, embedding_model)
        next_k_words_embeddings = get_embeddings_batch(next_k_words, tokenizer, embedding_model)

        # Calculate the cosine similarity between the current word and each of the next k words
        cosine_similarities = cosine_similarity(curr_word_embedding, next_k_words_embeddings)

        # Average the cosine similarities for the k next words
        avg_similarity = np.mean(cosine_similarities)
        similarities.append(avg_similarity)

    return np.mean(similarities)


# Function to calculate derailment metric
def calculate_coherence_metric(valid_answers_list, k, tokenizer, embedding_model):
    scores = []
    for sample_index, answers in enumerate(valid_answers_list):
        print(f"processing sample number {sample_index}...")
        if answers:
            for answer, question_index in answers:
                print(f"processing question number {question_index}...")
                cleaned_answer = clean_text(answer)
                score = calculate_answer_coherence_score(cleaned_answer, k, tokenizer, embedding_model)
                if score is not None:
                    scores.append({
                        'sample_index': sample_index,
                        'question_index': question_index,
                        'score': score
                    })
    return scores


# Function to calculate semantic similarity between k-grams in an answer using sliding window
def calculate_answer_coherence_score_sliding_window(answer, k, tokenizer, embedding_model):
    words = answer.split()
    similarities = []
    for i in range(len(words) - k):
        k_gram_1 = ' '.join(words[i:i + k])
        k_gram_2 = ' '.join(words[i + 1:i + k + 1])
        embeddings_1 = get_embeddings_batch([k_gram_1], tokenizer, embedding_model)
        embeddings_2 = get_embeddings_batch([k_gram_2], tokenizer, embedding_model)
        similarity = cosine_similarity(embeddings_1, embeddings_2)
        similarities.append(similarity[0][0])
    return np.mean(similarities)


# Function to calculate derailment metric using the sliding window approach
def calculate_coherence_score_sliding_window(valid_answers_list, k, tokenizer, embedding_model):
    scores = []
    for sample_index, answers in enumerate(valid_answers_list):
        print(f"processing sample number {sample_index}...")
        for answer, question_index in answers:
            print(f"processing question number {question_index}...")
            cleaned_answer = clean_text(answer)
            score = calculate_answer_coherence_score_sliding_window(cleaned_answer, k, tokenizer, embedding_model)
            scores.append({
                'sample_index': sample_index,
                'question_index': question_index,
                'score': score
            })
    return scores
