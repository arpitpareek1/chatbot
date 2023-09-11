import json
import spacy
from chat_bot_methods import aggregate_vectors, tokenize_and_vectorize
from chatbot_data_set import keyword_synonyms
import numpy as np

nlp = spacy.load("en_core_web_sm")

# Precompute word vectors for keywords
keyword_vectors = {item["keyword"]: nlp(item["keyword"]).vector for item in keyword_synonyms}


print(keyword_vectors)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        return super(NumpyEncoder, self).default(obj)

with open("keyword_vectors.json", "w") as json_file:
    json.dump(keyword_vectors, json_file, cls=NumpyEncoder)


# def convert_questions_to_vectors(dataset):
#     for item in dataset:
#         #change thsi key every time there is new type of data set please 
#         question = item
#         vectors = tokenize_and_vectorize(question)
#         item["vector"] = aggregate_vectors(vectors)
#     return dataset


# for item in keyword_vectors:
#     if "vector" in item and isinstance(item, np.ndarray):
#         item["vector"] = item["vector"].tolist()

# Define the file path where you want to save the dataset as a JSON file
# file_path = "dataset.json"

# Open the file for writing and save the dataset in JSON format
# with open(file_path, 'w') as json_file:
#     json.dump(keyword_vectors, json_file, indent=4)
