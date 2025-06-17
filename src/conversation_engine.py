import os
import json
import streamlit as st
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.global_settings import CONVERSATION_FILE, SCORES_FILE, INDEX_STORAGE
from src.prompts import CUSTOM_AGENT_SYSTEM_TEMPLATE
import chromadb


def load_chat_store():
    if os.path.exists(CONVERSATION_FILE) and os.path.getsize(CONVERSATION_FILE) > 0:
        try:
            with open(CONVERSATION_FILE, "r") as f:
                data = json.load(f)
            chat_store = ChatMessageHistory()
            for msg in data.get("messages", []):
                if msg["role"] == "user":
                    chat_store.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    chat_store.add_ai_message(msg["content"])
        except json.JSONDecodeError:
            chat_store = ChatMessageHistory()
    else:
        chat_store = ChatMessageHistory()
    return chat_store


@tool
def save_score(score: int, level: str, content: str, total_guess: str) -> None:
    """
    Save the user's mental health score and level to a file.

    Args:
        score (int): Numeric score representing the user's mental health.
        level (str): The user's mental health level ("kém", "trung bình", "bình thường", "tốt").
        content (str): Content describing the user's mental health.
        total_guess (str): Total guess of the user's mental health.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "Time": current_time,
        "Score": score,
        "Level": level,
        "Content": content,
        "Total guess": total_guess
    }

    # Initialize data as an empty list
    data = []
    try:
        with open(SCORES_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is invalid, keep data as an empty list
        pass

    data.append(new_entry)
    with open(SCORES_FILE, "w") as f:
        json.dump(data, f, indent=4)


def initialize_chatbot(chat_store):
    client = chromadb.PersistentClient(path=INDEX_STORAGE)

    collection = client.get_collection(name="vector")

    @tool
    def dsm5_query(query: str) -> str:
        """Cung cấp thông tin liên quan đến các bệnh tâm thần theo tiêu chuẩn DSM-5."""
        results = collection.query(
            query_texts=[query],
            n_results=5,
            query_embeddings=OpenAIEmbeddings().embed_query(query),
        )

        print(f"Query: {query}, Results: {results}")

        # Extract documents from the nested list
        documents = results['documents'][0] if results['documents'] else []
        return "\n".join(documents)

    # Danh sách các tools
    tools = [dsm5_query, save_score]

    # Tạo prompt với lịch sử hội thoại
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_AGENT_SYSTEM_TEMPLATE.format()),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Khởi tạo LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    # Khởi tạo agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor, chat_store


def save_chat_store(chat_store):
    messages = []
    for msg in chat_store.messages:
        if isinstance(msg, dict):
            role = "user" if msg.get("type") == "human" else "assistant"
            content = msg.get("content")
        else:
            role = "user" if msg.type == "human" else "assistant"
            content = msg.content
        messages.append({"role": role, "content": content})
    with open(CONVERSATION_FILE, "w") as f:
        json.dump({"messages": messages}, f, indent=4)


def run_chatbot(agent_executor, chat_store, user_input):
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_store.messages
    })
    # Thêm tin nhắn vào lịch sử
    chat_store.add_user_message(user_input)
    chat_store.add_ai_message(response["output"])
    # Lưu lịch sử
    save_chat_store(chat_store)
    return response["output"]


def display_messages(chat_store, container):
    with container:
        for msg in chat_store.messages:
            # Handle both dict and object cases
            if isinstance(msg, dict):
                msg_type = msg.get('type')
                content = msg.get('content')
            else:
                msg_type = msg.type
                content = msg.content

            if msg_type == "human":
                with st.chat_message("user"):
                    st.markdown(content)
            elif msg_type == "ai":
                with st.chat_message("assistant"):
                    st.markdown(content)
