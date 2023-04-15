import streamlit as st
from streamlit_chat import message

from inference import load_llm_model
from search_engine import get_news
from gen_embedding import text2embedding
from vector_database import result4search

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

st.set_page_config(layout="wide")


def generate_answer(root_path, prompt, history):
    # 加载模型
    tokenizer, model = load_llm_model(root_path, "ChatGLM-6B\\chatglm-6b-int4")

    with container:
        if len(history) > 0:
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(prompt, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            for response, history in model.stream_chat(tokenizer,
                                                       prompt,
                                                       history,
                                                       max_length=max_length,
                                                       top_p=top_p,
                                                       temperature=temperature
                                                       ):
                query, response = history[-1]
                st.write(response)
        return history


def button_reset_event():
    st.session_state["state"] = []


if __name__ == "__main__":
    model_root_path = "D:\\GitHub\\LLM-Weights\\"

    container = st.container()
    # chatbot logo and title
    st.image("main_page_logo.png", width=64)
    st.title("A Chatbot powered by ChatGLM-6b")

    max_length = st.sidebar.slider(
        'max_length', 0, 4096, 2048, step=1
    )
    top_p = st.sidebar.slider(
        'top_p', 0.0, 1.0, 0.6, step=0.01
    )
    temperature = st.sidebar.slider(
        'temperature', 0.0, 1.0, 0.95, step=0.01
    )

    st.session_state["state"] = []

    # create a prompt text for the text generation
    prompt_text = st.text_area(label="用户命令输入",
                               height=100,
                               placeholder="请在这儿输入您的命令")

    # set button
    col1, col2 = st.columns([0.1, 0.9], gap="small")

    with col1:
        button_send = st.button("send", key="generate_answer")
    with col2:
        button_reset = st.button("reset", on_click=button_reset_event())

    if button_send:
        # news from web search engine
        search_news = get_news(prompt_text)
        if (search_news is not None) and len(search_news[0]) >= 1:
            relevant_news = get_news(prompt_text)[0]["body"]
        else:
            relevant_news = ""
        # knowledge from database
        database_answer = result4search(text2embedding(prompt_text))[0]
        if database_answer is not None:
            relevant_answer = database_answer["response"]
        else:
            relevant_answer = ""
        prompt_text = "问题：" + prompt_text + "，请参考以下内容生成答案：" + relevant_news + "。" + relevant_answer
        with st.spinner("AI正在思考，请稍等........"):
            st.session_state["state"] = generate_answer(model_root_path,
                                                        prompt_text,
                                                        st.session_state["state"])
