from flask import Flask, jsonify, request, render_template

from chat_bot_methods import create_vector, find_answer , load_victors
from chatbot_data_set import product_info_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['user_input']
    bot_message = find_answer(user_message)
    return jsonify({'bot_message': bot_message})


if __name__ == '__main__':
    app.run(debug=True)
