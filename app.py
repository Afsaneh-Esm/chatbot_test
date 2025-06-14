import streamlit as st
import requests

st.title("🌌 تست اولیه چت‌بات کیهانی")
st.write("این نسخه‌ی تستی ساده‌شده است.")

try:
    res = requests.get("https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY")
    data = res.json()
    st.image(data.get("url", ""), caption=data.get("title", ""))
    st.write(data.get("explanation", ""))
except:
    st.error("نتونستیم اطلاعات NASA رو بگیریم.")
