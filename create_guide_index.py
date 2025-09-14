"""
guide-index를 새로 생성하는 스크립트
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

def create_guide_index():
    """guide-index 새로 생성"""
    
    # 1. Pinecone 연결
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    
    # 2. 문서 로드
    print("문서 로딩 중...")
    loader = Docx2txtLoader('./Guide.docx')
    documents = loader.load()
    print(f"문서 길이: {len(documents[0].page_content)} 문자")
    
    # 3. 텍스트 분할 (최적화된 설정)
    print("문서 분할 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 더 작은 청크
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    document_list = text_splitter.split_documents(documents)
    print(f"분할된 청크 수: {len(document_list)}")
    
    # 4. 각 청크 내용 확인
    print("\n=== 청크 내용 확인 ===")
    for i, doc in enumerate(document_list[:3]):  # 처음 3개만
        print(f"\n청크 {i+1}:")
        print(f"길이: {len(doc.page_content)} 문자")
        print(f"내용: {doc.page_content[:200]}...")
    
    # 5. 임베딩 생성
    print("\n임베딩 생성 중...")
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    
    # 6. guide-index 생성
    index_name = 'guide-index'
    print(f"\n{index_name} 생성 중...")
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )
    
    # 7. 문서 벡터화 및 저장
    print("문서 벡터화 및 저장 중...")
    database = PineconeVectorStore.from_documents(
        document_list, 
        embedding, 
        index_name=index_name
    )
    
    print(f"\n✅ guide-index 생성 완료!")
    print(f"청크 수: {len(document_list)}")
    
    return index_name

if __name__ == "__main__":
    create_guide_index()
