import streamlit as st
from sentence_transformers import SentenceTransformer

st.title("✅ تست لود شدن sentence-transformers")

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("این یک جمله‌ی آزمایشی است.")
st.success("مدل با موفقیت لود شد!")
st.write("Embedding:", embedding[:10])

