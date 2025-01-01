# Set up input html
from flask import Flask, request, render_template
app = Flask(__name__)
@app.route("/")
def my_form():
    return render_template('input_info.html')

# Import Libraries
from keras.models import model_from_json
import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


# The process to go through to set up tokenizer as in model training
data = pd.read_csv('emails_spam_normal.csv')
ham_msg = data[data.spam == 0]
spam_msg = data[data.spam == 1]
ham_msg = ham_msg.sample(n=len(spam_msg), random_state=42)
balanced_data = pd.concat([ham_msg, spam_msg], ignore_index=True)  # append was removed in pandas 2.0
balanced_data['text'] = balanced_data['text'].str.replace('Subject', '')
punctuations_list = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)  # replace nothing, delete characters from punctuations_list
    return text.translate(temp)
balanced_data['text'] = balanced_data['text'].apply(lambda x: remove_punctuations(x))
def remove_stopwords(text):
    stop_words = stopwords.words('english')

    imp_words = []

    # Storing the important words
    for word in str(text).split():
        word = word.lower()

        if word not in stop_words:
            imp_words.append(word)

    output = " ".join(imp_words)
    return output

balanced_data['text'] = balanced_data['text'].apply(lambda text: remove_stopwords(text))
train_X, test_X, train_Y, test_Y = train_test_split(balanced_data['text'],
                                                    balanced_data['spam'],
                                                    test_size = 0.2,
                                                    random_state = 42)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)

# Apply trained model for spam detection
@app.route('/', methods=['POST'])
def my_form_post():
    original_text = request.form['text']
    text = remove_punctuations(original_text)
    text = remove_stopwords(text)
    numeric_text = tokenizer.texts_to_sequences([text])
    json_file = open('spam_detection.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("spam_detection.weights.h5")
    
    spam_probability = loaded_model.predict(np.array(numeric_text))[0][0]
    if spam_probability > 0.8:
        processed_text = 'This is a spam email.'
    else:
        processed_text = 'This is a normal email.'
    return render_template('output_info.html', text = original_text, processed_text = processed_text)

if __name__ == '__main__':
    app.run(debug=True)
