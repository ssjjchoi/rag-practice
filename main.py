import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. PDF 읽기
def load_pdf(path):
    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


# PDF 로드
text = load_pdf("paper.pdf")


# 2. 텍스트 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_text(text)

print(f"Total Chunks: {len(chunks)}")


# 3. 임베딩 모델
model = SentenceTransformer('all-MiniLM-L6-v2')


# 4. 벡터 만들기
embeddings = model.encode(chunks)
embeddings = np.asarray(embeddings, dtype=np.float32)


# 5. FAISS 저장
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("RAG 준비 완료")


# 6. 질문 루프
while True:
    query = input("\n질문: ")

    if not query:
        continue

    if query.lower() == "exit":
        break

    q_emb = model.encode(
        [query],
        convert_to_numpy=True
    ).astype(np.float32)

    D, I = index.search(q_emb, k=3)

    print("\n[관련 문장]")

    for d, i in zip(D[0], I[0]):
        print(f"\nChunk #{i} | Distance: {d:.4f}")
        print(chunks[i])