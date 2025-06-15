import streamlit as st
from iris_classification import iris_app
from mnist_cnn_tf import mnist_app
from nlp_review_spacy import nlp_app

st.set_page_config(page_title="AI Toolkit  App", layout="centered")
st.title("AI Tools - Multi-Task Demo")

# Sidebar for navigation
task = st.sidebar.radio("Select a Task", [
    "🌸 Iris Classification",
    "🧠 MNIST Digit Classification",
    "📝 NLP - NER & Sentiment"
])

# Call appropriate module
if task == "🌸 Iris Classification":
    iris_app()

elif task == "🧠 MNIST Digit Classification":
    mnist_app()

elif task == "📝 NLP - NER & Sentiment":
    nlp_app()
