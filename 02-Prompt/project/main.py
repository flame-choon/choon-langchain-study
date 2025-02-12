import streamlit as st
import glob
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.prompts import load_prompt


# 세션에 채팅 메시지를 저장
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션에 저장 된 채팅 메시지를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 체인 생성
def create_chain(prompt_file_path, task=""):
    # 프롬프트
    prompt = load_prompt(prompt_file_path)
    if task:
        prompt = prompt.partial(task=task)  # task 에 task 값을 할당

    # GPT
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 출력 파서
    output_parser = StrOutputParser()

    # Chain 생성
    chain = prompt | llm | output_parser
    return chain


# .env 파일 로드
load_dotenv()

# langSmith에 로깅 할 프로젝트 명을 입력
logging.langsmith("02-Prompt")

st.title("나만의 ChatGPT")

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!!")

# 사이드 바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    prompt_files = glob.glob("prompts/*.yaml")
    selected_prompt = st.selectbox("프롬프트를 선택해 주세요", prompt_files, index=0)
    task_input = st.text_input("TASK 입력", "")

# 최초 1회에만 Session을 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []


if clear_btn:
    st.session_state["messages"] = []

# Session에 저장된 메시지 출력
print_messages()


# 사용자 입력이 들어올 시
if user_input:
    # 웹에 대화를 출력
    st.chat_message("user").write(user_input)
    # chain 을 생성
    chain = create_chain(selected_prompt, task=task_input)
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어 여기에 토큰을 스트리밍 출력한다
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
    # ai_answer = chain.invoke({"question": user_input})  # invoke로 호출하였기에 대답이 한번에 출력

    # GPT 의 답변
    # st.chat_message("assistant").write(ai_answer)

    # 대화기록을 session에 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
