import fitz  # pymupdf
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. PDF 읽기
def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 2. 문장 쪼개기
def split_text(text, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 3. 임베딩 모델
model = SentenceTransformer('all-MiniLM-L6-v2')

# PDF 로드
text = load_pdf("paper.pdf")
chunks = split_text(text)

# 4. 벡터 만들기
embeddings = model.encode(chunks)

# 5. FAISS 저장
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

print("RAG 준비 완료")

# 6. 질문 루프
while True:
    query = input("\n질문: ")
    q_emb = model.encode([query])

    D, I = index.search(np.array(q_emb), k=3)

    print("\n[관련 문장]")
    for i in I[0]:
        print("-", chunks[i])