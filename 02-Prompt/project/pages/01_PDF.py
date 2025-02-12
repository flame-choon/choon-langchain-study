import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# .env 파일 로드
load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 최초 1회에만 Session을 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# langSmith에 로깅 할 프로젝트 명을 입력
logging.langsmith("02-Prompt")

st.title("PDF 기반 QA")

warning_msg = st.empty()  # 경고용 메시지


# 세션에 채팅 메시지를 저장
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션에 저장 된 채팅 메시지를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 체인 생성
def create_chain(retriever, modelName):
    # # 프롬프트
    # prompt = load_prompt(prompt_file_path)

    # 6단계 : 프롬프트 생성 (Create prompt)
    # 프롬프트를 불러온다
    prompt = load_prompt("prompts/pdf-rag.yaml")

    # 7단계 : 언어모델 (LLM) 생성
    # 모델을 생성한다
    llm = ChatOpenAI(model_name=modelName, temperature=0.1)

    # 8단계 : 체인 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# 파일을 캐시 저장 (시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다....")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 1단계 : 문서 로드 (Load document)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 2단계 : 문서 분할 (Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 3단계 : 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 4단계 : DB 생성 및 저장
    # 벡터스토어를 생성
    vectorStore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 5단계 : 검색기 (Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성
    retriever = vectorStore.as_retriever()
    return retriever


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    selected_prompt = "prompt/pdf-rag.yaml"

    selected_model = st.selectbox(
        "LLM Models", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], index=0
    )

# Session에 저장된 메시지 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!!")

# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, selected_model)
    st.session_state["chain"] = chain
    print(chain)

# 사용자 입력이 들어올 시
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 웹에 대화를 출력
        st.chat_message("user").write(user_input)

        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어 여기에 토큰을 스트리밍 출력한다
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 session에 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")
