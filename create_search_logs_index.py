"""
검색 로그를 저장할 search-logs 인덱스를 생성하는 스크립트
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

def create_search_logs_index():
    """search-logs 인덱스 생성"""
    
    # Pinecone 연결
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    
    # 기존 인덱스 확인
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    print(f"기존 인덱스: {existing_indexes}")
    
    index_name = 'search-logs'
    
    if index_name in existing_indexes:
        print(f"⚠️ {index_name}이 이미 존재합니다.")
        return index_name
    
    # search-logs 인덱스 생성
    print(f"\n{index_name} 생성 중...")
    try:
        pc.create_index(
            name=index_name,
            dimension=3072,  # text-embedding-3-large의 차원
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
        print(f"✅ {index_name} 생성 완료!")
        return index_name
        
    except Exception as e:
        print(f"❌ 인덱스 생성 실패: {e}")
        return None

if __name__ == "__main__":
    create_search_logs_index()
