import re
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from settings import Settings

@dataclass
class Chunk:
    board_id: int
    title: str
    author: str
    text: str
    score: float = 0.0

# 긴 게시글을 나누는 함수 (청킹), chunk_size=500, chunk_overlap=80
# RecursiveCharacterTextSplitter: LangChain에서 제공하는 텍스트 분할 도구
def make_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    # \n\n → \n → ". " → 공백 순서로 자연스러운 경계를 찾아 자름
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

# 청크마다 제목을 붙임
def chunk_texts(splitter: RecursiveCharacterTextSplitter, title: str, content: str) -> list[str]:
    """청크마다 제목을 붙여 둡니다. 본문만 있는 청크도 어떤 글인지 검색될 수 있게 하기 위함."""
    pieces = splitter.split_text(content) if content.strip() else []  # 분문 분할
    if not pieces:  # 분분이 비어 있는 경우
        pieces = [""]
    return [f"제목: {title}\n{p}".strip() for p in pieces]

# 검색 결과에서 본문만 추출
def body_of(text: str) -> str:
    """청크 앞의 '제목:' 줄을 뗀 본문 (출처 미리보기용)."""
    lines = text.split("\n", 1)
    return lines[1].strip() if len(lines) > 1 and lines[0].startswith("제목:") else text.strip()

# 글자를 두 갰기 묶기: 실제 임베딩 모델을 사용하는 대인, 문자열을 2-gram으로 변환
def bigrams(text: str) -> set[str]:
    """글자 2개 묶음. 공백·기호는 빼고 소문자로."""
    s = re.sub(r"[^0-9a-zA-Z가-힣]", "", text.lower())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else ({s} if s else set())


# RAG 엔진을 대신하는 임시 클래스
class StubEngine:
    name = "stub"
    min_score = 0.0

    def __init__(self, settings):  # 생성자
        self.store = {}

    # 게시글 저장
    def index(self, board_id, title, content, author, email):
        self.store[board_id] = [content]
        return 1

    # 게시글 삭제
    def delete(self, board_id):
        return len(self.store.pop(board_id, []))

# 엔진 생성 함수
def create_engine(settings):
    return StubEngine(settings)
