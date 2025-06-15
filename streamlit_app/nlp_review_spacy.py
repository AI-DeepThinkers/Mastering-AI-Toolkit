# streamlit_app/nlp.py

import spacy
from textblob import TextBlob
import streamlit as st

def nlp_app():
    st.subheader("📝 NLP: NER & Sentiment Analysis")
    text = st.text_area("Enter a review or product comment:")

    if text:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        st.markdown("### 🔍 Named Entities")
        if doc.ents:
            for ent in doc.ents:
                st.write(f"• **{ent.text}** → `{ent.label_}`")
        else:
            st.write("No named entities found.")

        sentiment = TextBlob(text).sentiment
        st.markdown("### 💬 Sentiment")
        st.write(f"Polarity: `{sentiment.polarity:.2f}`")
        st.write(f"Subjectivity: `{sentiment.subjectivity:.2f}`")

        label = (
            "Positive 😊" if sentiment.polarity > 0 else
            "Negative 😠" if sentiment.polarity < 0 else
            "Neutral 😐"
        )
        st.success(f"Overall Sentiment: **{label}**")


# Optional demo test code (for CLI use only)
if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    review = "I absolutely love the sound quality of my new Bose headphones!"

    print("=" * 60)
    print("📝 Review Text:\n", review)
    print("=" * 60)

    doc = nlp(review)
    print("\n🔍 Named Entities Found:")
    if doc.ents:
        for ent in doc.ents:
            print(f"• {ent.text:<25} ({ent.label_})")
    else:
        print("• No named entities detected.")

    sentiment = TextBlob(review).sentiment
    print("\n💬 Sentiment Analysis:")
    print(f"• Polarity    : {sentiment.polarity:.3f}")
    print(f"• Subjectivity: {sentiment.subjectivity:.3f}")

    sentiment_label = (
        "Positive" if sentiment.polarity > 0 else
        "Negative" if sentiment.polarity < 0 else
        "Neutral"
    )
    print(f"• Overall Sentiment: {sentiment_label}")
    print("=" * 60)
