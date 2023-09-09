import spacy
from chatbot_data_set import chatbot_dataset, product_faqs
from gensim.models import Word2Vec
import re


# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


dataset = chatbot_dataset


def find_answer(user_question):
    if "product" in user_question:
        isprodt = calculate_product_similarity(user_question)
        if isprodt is not None:
            return isprodt

    max_similarity = 0
    best_match = None

    similer_que = []
    for item in dataset:
        # Calculate similarity using custom vectors
        similarity = calculate_similarity_with_custom_vectors(user_question, item["que"])
        print("similarity", similarity)

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = item["ans"]
            if similarity > 0.45 and similarity < 0.8:
                similer_que.append(item['que'])


    if max_similarity >= 0.65:
        return {"ans":best_match ,"ques":similer_que}
    else:
        if len(user_question) < 15:
            isGreet = calculate_greeting_similarity(user_question)

            if isGreet is None:
                return {"ans":"I'm sorry, but it looks like you pressed enter without fully writing your question. Please try again." ,"ques":similer_que}
            elif isGreet:
               return {"ans":isGreet ,"ques":similer_que}
               
        else:
            return {"ans":"Sorry, I didn't get you." ,"ques":similer_que}
             

def calculate_greeting_similarity(user_question):
    greetings = ["hello", "hi", "hey", "good morning",
                 "good afternoon", "good evening"]

    similarities = [nlp(user_question).similarity(nlp(greeting))
                    for greeting in greetings]

    max_similarity = max(similarities)

    if max_similarity >= 0.7:
        best_greeting = greetings[similarities.index(max_similarity)]
        return f"{best_greeting.capitalize()}! How can I assist you today?"

    return None


def calculate_product_similarity(user_question):

    similarities = [nlp(user_question).similarity(
        nlp(product_faq['que']))for product_faq in product_faqs]

    max_similarity = max(similarities)

    if max_similarity >= 0.7:
        best_greeting = product_faqs[similarities.index(max_similarity)]
        return best_greeting

    return None


def create_vector(provided_text):
    print("creating vector")
    sentences = re.split(r'(?<=[.!?])\s+', provided_text)
    tokenized_sentences = [sentence.split() for sentence in sentences]
    model = Word2Vec(sentences=tokenized_sentences,
                     vector_size=100, window=5, min_count=1, sg=0)
    model.wv.save_word2vec_format('vectors.txt', binary=False)


def load_victors(path):
    nlp.vocab.vectors.from_disk(path)


def calculate_similarity_with_custom_vectors(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    # Calculate similarity using custom vectors
    similarity = doc1.similarity(doc2)
    return similarity
