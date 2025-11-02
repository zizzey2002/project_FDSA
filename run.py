# import uuid
# import chromadb
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# embedding_model = OpenAIEmbeddingFunction(model_name="text-embedding-v4",
#                                           api_key="sk-dfb5d9295ee94adeafc438d18c7d4900",
#                                           api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
# client = chromadb.PersistentClient(path="./chroma_db")

# #check collection if exists
# collection_names = [c.name for c in client.list_collections()]
# if "policies" in collection_names:
#     collection = client.get_collection("policies", embedding_function=embedding_model)
# else:
#     collection = client.create_collection(name="policies", embedding_function=embedding_model)

# policies = [
#     "The policy is to deny all requests.",
#     "The policy is to allow all requests.",
#     "The policy is to allow requests from trusted sources.",
#     "The policy is to deny requests from untrusted sources.",
#     "The policy is to allow requests from trusted sources and deny requests from untrusted sources.",
#     "The policy is to allow requests from trusted sources and deny requests from untrusted sources and log all requests.",
#     "The policy is to allow requests from trusted sources and deny requests from untrusted sources and log all requests"
#     ]

# collection.add(
#     ids=[str(uuid.uuid4()) for _ in policies],
#     documents=policies,
#     metadatas=[{"line": line} for line in range(len(policies))]
# )


# results = collection.query(
#     query_texts=["The policy is to deny all requests it."], # Chroma will embed this for you
#     n_results=2 # how many results to return
# )
# print(results)






# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# embedding_model = OpenAIEmbeddingFunction(model_name="text-embedding-v4",
#                                           api_key="sk-dfb5d9295ee94adeafc438d18c7d4900",
#                                           api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")

# from chromadb import PersistentClient
# client = PersistentClient(path="./chroma_db")
# # 初始化客户端、

# # 获取 collection
# collection = client.get_collection("policies", embedding_function=embedding_model)

# # 查看全部数据
# data = collection.get(include=["embeddings", "documents", "metadatas"])
# print(data["embeddings"][:2])
# print(data["documents"][:2])
# print(data["metadatas"][:2])







from docx import Document
import uuid
def read_docx(documents:str, chunk_size=1000):
    # read file
    doc = Document(documents)
    texts = []
    for p in doc.paragraphs:
        if p.text.strip():
            # print(p.text.strip())
            # print("-"*50)
            texts.append(p.text.strip())
    # print("\n".join(texts))
    return "\n".join(texts)

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_id = str(uuid.uuid4())
        chunk = text[start:end]
        chunks.append({"chunk_id": chunk_id, "text": chunk})
        start += chunk_size - overlap
    return chunks

def segment_chunk(chunk, max_len=200):
    # 按句号或换行符再细分
    import re
    sentences = re.split(r'(?<=[。！？.!?])', chunk)
    segments, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_len:
            current += s
        else:
            if current:
                segments.append(current.strip())
            current = s
    if current:
        segments.append(current.strip())
    return segments


text = read_docx("./中华人民共和国刑法_20201226.docx")
chunks = chunk_text(text)
# for i in range(len(chunks)):
#     print(f"chunk {i}: {chunks[i]}")
    
chunk_segments = []
for c in chunks:
    segs = segment_chunk(c["text"])
    for seg in segs:
        seg_id = str(uuid.uuid4())
        chunk_segments.append({
            "chunk_id": c["chunk_id"],
            "segment_id": seg_id,
            "text": seg
        })
        print(f"chunk {c['chunk_id']} segment {seg_id}:")
        print(seg)