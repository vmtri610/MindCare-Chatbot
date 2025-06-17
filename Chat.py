import streamlit as st
import os
from dotenv import load_dotenv
from src.conversation_engine import initialize_chatbot, load_chat_store, run_chatbot, display_messages
from src.global_settings import CONVERSATION_FILE

load_dotenv()


def clear_chat_history():
    """Xóa lịch sử hội thoại trong chat_store và CONVERSATION_FILE."""
    # Xóa chat_store trong session_state
    if "chat_store" in st.session_state:
        st.session_state.chat_store = load_chat_store()  # Tạo chat_store mới, rỗng
    # Xóa CONVERSATION_FILE
    if os.path.exists(CONVERSATION_FILE):
        try:
            os.remove(CONVERSATION_FILE)
            st.success("Đã xóa lịch sử hội thoại!")
        except Exception as e:
            st.error(f"Lỗi khi xóa file: {e}")
    # Đặt lại trạng thái khởi tạo chatbot để hiển thị câu chào mới
    st.session_state.chatbot_initialized = False
    # Xóa nội dung chat_container bằng cách reruns ứng dụng
    st.rerun()


st.title("MindCare Chatbot")

# Cấu hình API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Khởi tạo session state
if "chat_store" not in st.session_state:
    st.session_state.chat_store = load_chat_store()
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "chatbot_initialized" not in st.session_state:
    st.session_state.chatbot_initialized = False

# Nút xóa lịch sử hội thoại
if st.button("Clear Chat History"):
    clear_chat_history()

# Container cho lịch sử hội thoại
chat_container = st.container()

# Tự động khởi tạo chatbot nếu chưa khởi tạo
if not st.session_state.chatbot_initialized:
    st.session_state.agent_executor, st.session_state.chat_store = initialize_chatbot(
        st.session_state.chat_store
    )
    st.session_state.chatbot_initialized = True
    # Hiển thị câu chào từ chatbot và lưu vào chat_store
    welcome_message = "Chào bạn! Mình là chatbot hỗ trợ sức khỏe tâm thần, luôn ở đây để lắng nghe và trò chuyện cùng bạn. Hôm nay bạn cảm thấy thế nào? Bạn muốn chia sẻ điều gì đang xảy ra với mình hôm nay không?"
    st.session_state.chat_store.add_ai_message(welcome_message)

# Hiển thị lịch sử hội thoại
with chat_container:
    display_messages(st.session_state.chat_store, chat_container)

# Nhập câu hỏi
user_input = st.chat_input("Nhập câu hỏi hoặc chia sẻ của bạn...")

# Xử lý câu hỏi
if user_input and st.session_state.agent_executor:
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Đang xử lý..."):
            response = run_chatbot(
                st.session_state.agent_executor,
                st.session_state.chat_store,
                user_input,
            )
        with st.chat_message("assistant"):
            st.markdown(response)
