import spacy
from flask import request, jsonify
from chatbot_data_set import chatbot_dataset, product_faqs
from dataset import data_set
from sklearn.metrics.pairwise import cosine_similarity


