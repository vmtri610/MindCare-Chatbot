import streamlit as st

home_page = st.Page("Home.py", title="Home", icon=":material/home:")
user_page = st.Page("User.py", title="User", icon=":material/analytics:")
chat_page = st.Page("Chat.py", title="Chat", icon=":material/chat:")

pg = st.navigation([
    home_page,
    user_page,
    chat_page,
])

st.set_page_config(
    page_title="MindCare Chatbot",
    page_icon="🧠",
    layout="wide",
)
pg.run()
