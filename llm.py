import warnings
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
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

# 환경 변수 로드
load_dotenv()

# Few-shot 예시는 현재 사용하지 않음
store = {}

# Pinecone 클라이언트 초기화 (검색 로그용)
from pinecone import Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
search_log_index = pc.Index("search-logs") if "search-logs" in [idx.name for idx in pc.list_indexes()] else None


def log_search_query(query: str, results_count: int, top_scores: list):
    """검색 쿼리를 Pinecone에 로그로 저장합니다."""
    try:
        if search_log_index is None:
            print("⚠️ search-logs 인덱스가 없습니다. 로그 저장을 건너뜁니다.")
            return
        
        # 검색 로그 데이터 생성
        log_id = str(uuid.uuid4())
        log_data = {
            "query": query,
            "results_count": results_count,
            "top_score_1": top_scores[0] if len(top_scores) > 0 else 0.0,
            "top_score_2": top_scores[1] if len(top_scores) > 1 else 0.0,
            "top_score_3": top_scores[2] if len(top_scores) > 2 else 0.0,
            "timestamp": datetime.now().isoformat(),
            "session_id": "abc123"
        }
        
        # 쿼리를 임베딩으로 변환
        from langchain_openai import OpenAIEmbeddings
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        query_embedding = embedding.embed_query(query)
        
        # Pinecone에 로그 저장
        search_log_index.upsert(vectors=[{
            "id": log_id,
            "values": query_embedding,
            "metadata": log_data
        }])
        
        print(f"✅ Search log saved: {query[:50]}...")
        
    except Exception as e:
        print(f"❌ Search log save failed: {e}")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID에 해당하는 채팅 기록을 가져오거나 새로 생성합니다."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


class LoggingRetriever:
    """검색 로그를 저장하는 커스텀 retriever"""
    def __init__(self, retriever):
        self.retriever = retriever
    
    def get_relevant_documents(self, query):
        # 원본 검색 실행
        docs = self.retriever.get_relevant_documents(query)
        
        # 검색 결과에서 점수 추출
        top_scores = []
        for doc in docs:
            if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                top_scores.append(doc.metadata['score'])
            else:
                top_scores.append(0.0)
        
        # 검색 로그 저장
        log_search_query(query, len(docs), top_scores[:3])  # 상위 3개 점수만 저장
        
        return docs
    
    def invoke(self, input_data, config=None):
        """LangChain 호환성을 위한 invoke 메서드"""
        if isinstance(input_data, dict):
            query = input_data.get("input", input_data.get("query", ""))
        else:
            query = str(input_data)
        
        docs = self.get_relevant_documents(query)
        return docs
    
    def __or__(self, other):
        """LangChain 호환성을 위한 OR 연산자"""
        return self

def get_retriever():
    """
    Pinecone에 'guide-index-optimized'가 이미 존재한다고 가정하고 retriever를 반환합니다.
    """
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'guide-index'  # guide-index 사용
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    
    # 검색 파라미터 최적화 (점수 포함)
    retriever = database.as_retriever(
        search_type="similarity_score_threshold",  # 점수 기반 검색
        search_kwargs={
            'k': 5,  # 최적화된 문서 수
            'score_threshold': 0.3  # 관련성 높은 문서만 사용
        }
    )
    
    return retriever


def get_history_retriever():
    """채팅 기록을 고려하여 검색 질문을 재구성하는 retriever를 생성합니다."""
    llm = get_llm()
    base_retriever = get_retriever()
    
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
        llm, base_retriever, contextualize_q_prompt
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
    
    # Few-shot 예시 정의
    few_shot_examples = [
        {
            "input": "What is the current ratio?",
            "output": "The current ratio is calculated as Current Assets / Current Liabilities. It measures a company's ability to pay short-term obligations. A ratio above 1.0 indicates the company has more current assets than current liabilities."
        },
        {
            "input": "How is ROE calculated?",
            "output": "ROE (Return on Equity) is calculated as Net Income / Shareholders' Equity. It measures how efficiently a company uses shareholders' equity to generate profits. A higher ROE indicates better efficiency in using equity capital."
        },
        {
            "input": "What is the quick ratio?",
            "output": "The quick ratio, or acid-test ratio, is calculated as (Current Assets - Inventories) / Current Liabilities. It measures a company's ability to meet short-term liabilities without relying on inventory sales. A ratio above 1.0 is generally considered healthy."
        },
        {
            "input": "How do you calculate net profit margin?",
            "output": "Net profit margin is calculated as Net Income / Revenue. It indicates how much net income is generated as a percentage of revenue. A higher percentage indicates better profitability and operational efficiency."
        },
        {
            "input": "What is the VIX?",
            "output": "The VIX, or Volatility Index, is a key indicator of market uncertainty and risk, measuring the market's expectations of volatility over the coming 30 days. It's often called the 'fear gauge' of the market."
        }
    ]
    
    # Few-shot 프롬프트 템플릿
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ]),
        examples=few_shot_examples
    )
    
    # 개선된 시스템 프롬프트 (Chain of Thought 포함)
    system_prompt = (
        "You are a senior financial analyst at Unity Financial Group with 10+ years of experience. "
        "Your expertise includes financial ratios, risk management, regulatory compliance, and market analysis.\n\n"
        
        "Answer Guidelines:\n"
        "1. Think step by step before answering:\n"
        "   - Identify the key financial concept being asked\n"
        "   - Find relevant information in the provided context\n"
        "   - Apply appropriate financial principles and formulas\n"
        "   - Provide a clear, structured answer\n\n"
        
        "2. Answer Structure:\n"
        "   - Start with a clear definition or explanation\n"
        "   - Include the specific formula when applicable\n"
        "   - Provide interpretation and significance\n"
        "   - Use professional financial terminology\n\n"
        
        "3. Quality Standards:\n"
        "   - Use ONLY information from the provided context\n"
        "   - Include specific numbers, formulas, and calculations when relevant\n"
        "   - If information is not in context, explicitly state 'Not available in provided context'\n"
        "   - Maintain accuracy and professionalism\n\n"
        
        "4. Response Format:\n"
        "   - Be concise but comprehensive (2-4 sentences for simple questions)\n"
        "   - Use bullet points only when listing multiple related items\n"
        "   - Include relevant examples when helpful\n\n"
        
        "Context: {context}\n"
        "Question: {input}\n"
        "Step-by-step reasoning:"
    )
    
    # 최종 프롬프트 구성
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
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
    # 검색 로그 저장을 위한 직접 검색 (Pinecone API 직접 사용)
    try:
        from langchain_openai import OpenAIEmbeddings
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        query_embedding = embedding.embed_query(user_message)
        
        # Pinecone에서 직접 검색하여 점수 가져오기
        guide_index = pc.Index("guide-index")
        search_results = guide_index.query(
            vector=query_embedding,
            top_k=6,
            include_metadata=True
        )
        
        # 검색 결과에서 점수 추출
        top_scores = []
        for match in search_results.matches:
            top_scores.append(match.score)
        
        print(f"Retrieved documents: {len(search_results.matches)}")
        print(f"Top scores: {top_scores[:3]}")
        
        # 검색 로그 저장
        log_search_query(user_message, len(search_results.matches), top_scores[:3])
        
    except Exception as e:
        print(f"⚠️ 검색 로그 저장 실패: {e}")
    
    # RAG 체인 실행
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