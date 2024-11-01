import streamlit as st
import pandas as pd
import re
import tensorflow as tf
import tensorflow_hub as tf_hub
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import nltk

# Download the stopwords resource
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load the trained model
model = load_model('model.h5', custom_objects={'KerasLayer': tf_hub.KerasLayer})
# Load Dataset
dataset = 'tripadvisor_hotel_reviews.csv'
data = pd.read_csv(dataset)

# Load stopwords
stpwds_en = list(set(stopwords.words('english')))

# Text preprocessing function
def text_preprocessing(text):
    # Case folding
    text = text.lower()

    # Mention removal
    text = re.sub("@[A-Za-z0-9_]+", " ", text)

    # Hashtags removal
    text = re.sub("#[A-Za-z0-9_]+", " ", text)

    # Newline removal (\n)
    text = re.sub(r"\\n", " ",text)

    # Whitespace removal
    text = text.strip()

    # URL removal
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www.\S+", " ", text)

    # Non-letter removal (such as emoticons, symbols, etc.)
    text = re.sub("[^A-Za-z\s']", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords removal
    tokens = [word for word in tokens if word not in stpwds_en]

    # Combining Tokens
    text = ' '.join(tokens)

    return text

# Define the Streamlit interface
st.title('Sentiment Analysis for Hotel Reviews')

# Image
image_path = 'image.png'  
st.image(image_path, width=700)  

# Display the pie chart for ratings
if st.checkbox('Show Ratings Distribution'):
    # Count the occurrences of each rating
    rating_counts = data['Rating'].value_counts()
    
    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
    plt.title('Ratings Distribution')
    
    # Display the pie chart in Streamlit
    st.pyplot(fig)

    # Description
    conclusion_text = """
    From this graph, we can conclude:

    - Negative ratings (1-2) have approximately 3000 data points.
    - Neutral ratings (3) have around 2000 data points.
    - Positive ratings (4-5) have approximately 15000 data points.

    Conclusion:
    Positive ratings are more dominant than negative ratings, indicating that the data is not well-distributed or balanced. This can affect the model's performance when predicting ratings from reviews given by users or customers. Therefore, we need to perform data balancing.
    """

    # In your Streamlit app, you can display this text using st.markdown()
    st.markdown(conclusion_text)

# Get user input
user_input = st.text_area("Enter the text for sentiment analysis:")

if st.button('Analyze'):
    if user_input:
        # Preprocess the input text
        processed_text = text_preprocessing(user_input)
        prediction = model.predict([[processed_text]])
        sentiment = "Positive" if prediction[0][1] > 0.5 else "Negative"

        if sentiment == "Positive":
            
            image_path = 'thumbs_up.png'  
            st.image(image_path, width=200)  

            # Display the result
            st.write(f"Sentiment: {sentiment}")
        else:
            image_path = 'thumbs_down.png'  
            st.image(image_path, width=200)  

            # Display the result
            st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter some text.")