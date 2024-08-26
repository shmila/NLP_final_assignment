import re

import numpy as np
import spacy_stanza
import torch


# Function to clean and normalize text
def clean_text(text):
    text = text.lower()  # Normalize case
    text = re.sub(r'\+\+[\w\s]+?\+\+', '', text)  # Remove image titles and question titles
    text = re.sub(r'[^א-ת\s]', '', text)  # Remove non-Hebrew characters
    text = re.sub(r'[a-zA-Z]', '', text)  # Remove English transliterations
    text = ' '.join(text.split())  # Remove extra whitespace
    text = remove_repeated_phrases(text)
    return text


# Function to preprocess text using SpaCy
def preprocess_text(text, nlp):
    doc = nlp(text)
    tokens = [{'text': token.text, 'lemma': token.lemma_, 'pos': token.pos_, 'dep': token.dep_} for token in doc]
    return tokens


def get_embeddings_batch(sentences, tokenizer, embedding_model):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(embedding_model.device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


def extract_open_answers(transcription):
    questions = [r'\+\+בר מצווה\+\+', r'\+\+מה אוהב לעשות\+\+', r'\+\+מה מעצבן\+\+', r'\+\+משאלות לעתיד\+\+']
    answers = []
    for i in range(len(questions) - 1):
        pattern = re.compile(f'{questions[i]}(.*?){questions[i + 1]}', re.DOTALL)
        match = pattern.search(transcription)
        if match:
            answers.append((match.group(1).strip(), i))
    # Extract the last answer separately
    last_pattern = re.compile(f'{questions[-1]}(.*)', re.DOTALL)
    last_match = last_pattern.search(transcription)
    if last_match:
        answers.append((last_match.group(1).strip(), len(questions) - 1))
    return answers


def extract_image_descriptions(transcription, num_image_description_questions=14):
    image_questions = [r'\+\+' + f'תמונה {i}' + r'\+\+' for i in 1 + np.arange(num_image_description_questions)]
    descriptions = []

    for i in range(len(image_questions) - 1):
        pattern = re.compile(f'{image_questions[i]}(.*?){image_questions[i + 1]}', re.DOTALL)
        match = pattern.search(transcription)
        if match:
            descriptions.append((match.group(1).strip(), i + 1))

    # Extract the last image description separately
    last_pattern = re.compile(f'{image_questions[-1]}(.*)', re.DOTALL)
    last_match = last_pattern.search(transcription)
    if last_match:
        descriptions.append((last_match.group(1).strip(), len(image_questions)))

    return descriptions


# Function to extract answers to the four open questions
def extract_valid_answers(transcriptions, min_word_count=50):
    valid_answers = []
    for transcription in transcriptions:
        answers = extract_open_answers(transcription)
        valid_answers.append(
            [(clean_text(answer), index) for answer, index in answers if
             len(clean_text(answer).split()) > min_word_count])
    return valid_answers


# Function to extract answers to the four open questions
def extract_valid_image_descriptions(transcriptions, min_word_count=50):
    valid_descriptions = []
    for idx, transcription in enumerate(transcriptions):
        answers = extract_image_descriptions(transcription)
        valid_descriptions.append(
            [(clean_text(answer), index) for (answer, index) in answers if
             len(clean_text(answer).split()) > min_word_count])
    return valid_descriptions


def remove_repeated_phrases(text, max_n=5):
    """Remove repeated phrases from text using n-gram analysis."""
    words = text.split()
    length = len(words)

    # Set to store the starting index of the phrases already removed to avoid overlapping
    removed_indices = set()

    for n in range(2, max_n + 1):
        i = 0
        while i < length - n + 1:
            ngram = ' '.join(words[i:i + n])
            next_index = i + n
            # Check if the next n words are the same
            if next_index + n <= length and ' '.join(words[next_index:next_index + n]) == ngram:
                # Remove one instance
                del words[i:next_index]
                length -= n
                removed_indices.add(i)
            else:
                i += 1

    return ' '.join(words)


def extract_content_words(text):
    nlp = spacy_stanza.load_pipeline('he')
    doc = nlp(text)
    content_words = [token.text for token in doc if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}]
    return ' '.join(content_words)


def extract_content_words_without_PROPN(text):
    nlp = spacy_stanza.load_pipeline('he')
    doc = nlp(text)
    content_words = [token.text for token in doc if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}]
    return ' '.join(content_words)
