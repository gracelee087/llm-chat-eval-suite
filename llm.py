import warnings
# Pydantic v2 경고 무시
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_pinecone")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Few-shot 예시는 현재 사용하지 않음
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID에 해당하는 채팅 기록을 가져오거나 새로 생성합니다."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    """
    Pinecone에 'guide-index-optimized'가 이미 존재한다고 가정하고 retriever를 반환합니다.
    """
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'guide-index'  # guide-index 사용
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    
    # 검색 파라미터 최적화
    retriever = database.as_retriever(
        search_type="similarity_score_threshold",  # 점수 기반 검색
        search_kwargs={
            'k': 6,  # 더 많은 문서 검색 (4 -> 6)
            'score_threshold': 0.7,  # 점수 임계값 설정
            'include_metadata': True  # 메타데이터 포함
        }
    )
    return retriever


def get_history_retriever():
    """채팅 기록을 고려하여 검색 질문을 재구성하는 retriever를 생성합니다."""
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is. "
        "Make sure to include relevant keywords for better document retrieval."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_llm(model='gpt-4o'):
    """주요 LLM 모델 객체를 생성합니다."""
    llm = ChatOpenAI(model=model)
    return llm


def get_rag_chain():
    """
    RAG 체인 및 프롬프트를 재구성합니다.
    """
    llm = get_llm()
    
    system_prompt = (
        "You are an expert financial analyst. Answer the user's questions about Financial Reporting Standards and Employee Handbook."
        "Please use the provided document to answer the question, and if you cannot find the answer, just say you don't know."
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    
    return conversational_rag_chain


def get_ai_response(user_message):
    """사용자의 메시지에 대한 AI의 답변을 스트리밍 방식으로 반환합니다."""
    rag_chain = get_rag_chain()
    
    ai_response = rag_chain.stream(
        {
            "input": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response