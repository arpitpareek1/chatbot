import spacy
from chatbot_data_set import greetings, product_faqs , keyword_synonyms
from dataset import data_set
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_sm")



def find_fixed_keyword(user_input):
    for fixed_keyword, synonyms in keyword_synonyms.items():
        if user_input in synonyms:
            find_synonym(user_input, fixed_keyword)
            return fixed_keyword
    return None 


def find_synonym(user_input, fixed_keyword):
    user_doc = nlp(user_input)
    fixed_keyword_doc = nlp(fixed_keyword)
    
    similarity = user_doc.similarity(fixed_keyword_doc)
    
    # You can adjust the similarity threshold as needed
    if similarity > 0.7:
        return fixed_keyword
    else:
        return user_input

def find_answer(user_que):
    user_question = user_que.lower()
    if "product" in user_question:
        isprodt = calculate_product_similarity(user_question)
        if isprodt is not None:
            return isprodt

        
    #new aproch
    best_match = find_best_match(user_question, data_set)
    if best_match:
        return best_match
    else :
        return {"ans":"Sorry, I didn't get you." ,"ques":[]}

             

def calculate_greeting_similarity(user_question):
    similarities = [nlp(user_question).similarity(nlp(greeting))
                    for greeting in greetings]

    max_similarity = max(similarities)

    if max_similarity >= 0.7:
        best_greeting = greetings[similarities.index(max_similarity)]
        return f"{best_greeting.capitalize()}!"

    return None

def calculate_product_similarity(user_question):

    similarities = [nlp(user_question).similarity(
        nlp(product_faq['que']))for product_faq in product_faqs]

    max_similarity = max(similarities)

    if max_similarity >= 0.7:
        best_greeting = product_faqs[similarities.index(max_similarity)]
        return best_greeting

    return None

def tokenize_and_vectorize(text):
    doc = nlp(text)
    vectors = [token.vector for token in doc]
    return vectors

def aggregate_vectors(vectors):
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return None

def calculate_similarity(user_vector, dataset_vectors):
    similarities = cosine_similarity([user_vector], dataset_vectors)
    return similarities[0]

def find_best_match(user_que, dataset):
    ques = []
    ans = ""

    user_vector = aggregate_vectors(tokenize_and_vectorize(user_que))
    if user_vector is not None:
        similarities = calculate_similarity(
            user_vector, [item["vector"] for item in dataset])
        
        best_match_index = similarities.argmax()
        print("similarities[best_match_index]",similarities[best_match_index])
        if similarities[best_match_index] > 0.8:
            ans = dataset[best_match_index]['ans']
        elif similarities[best_match_index] > 0.5:
            ans = "I think you are asking about " +  dataset[best_match_index]['que'] + " if yes then "+ dataset[best_match_index]['ans']
        else :
            if len(user_que) > 10:
                ans = "I didn't get you"
            else :
                isGreet = calculate_greeting_similarity(user_que)
                if isGreet is not None:
                    ans =  isGreet
                else:
                    ans = "Please write few details more"
        for index, item in enumerate(similarities):
            if item > 0.6 and index != best_match_index:
                ques.append(dataset[index]['que'])

        return {"ans":ans, "ques": ques}
    else:
        return {"ans":"I didn't get you", "ques": ques}
