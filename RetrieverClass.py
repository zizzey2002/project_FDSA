from docx import Document
import uuid
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import logging        
import pandas as pd
logging.basicConfig(level=logging.INFO)
import os
api_key = os.getenv("ALIYUN_API_KEY")

class Retriever:
    def __init__(self, file_path, db_path="./chroma_db", chunk_size=1000, overlap=200, segment_size=200):
        self.file_path = file_path
        self.file_name = file_path.split("/")[-1].split(".")[0]
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.segment_size = segment_size

        embedding_model = OpenAIEmbeddingFunction(
            model_name="text-embedding-v4",
            api_key=api_key,
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.client = chromadb.PersistentClient(path=db_path)
        #check collection if exists
        collection_names = [c.name for c in self.client.list_collections()]
        self.collection_name = str(uuid.uuid5(uuid.NAMESPACE_DNS, self.file_name))
        if self.collection_name in collection_names:
            logging.info(f"Collection {self.collection_name} already exists.")
            self.collection = self.client.get_collection(self.collection_name, embedding_function=embedding_model)
        else:
            self.collection = self.client.create_collection(name=self.collection_name, embedding_function=embedding_model, metadata={"hnsw:space": "cosine"})
            self.get_embedding()

    def read_docx(self, documents:str):
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

    def chunk_text(self, text, chunk_size=1000, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_id = str(uuid.uuid4())
            chunk = text[start:end]
            chunks.append({"chunk_id": chunk_id, "text": chunk})
            start += chunk_size - overlap
        return chunks

    def segment_chunk(self, chunk):
        # 按句号或换行符再细分
        import re
        sentences = re.split(r'(?<=[。！？.!?])', chunk)
        segments, current = [], ""
        max_len = self.segment_size
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

    def split(self):
        text = self.read_docx(self.file_path)
        chunks = self.chunk_text(text)
        # for i in range(len(chunks)):
        #     print(f"chunk {i}: {chunks[i]}")
            
        chunk_segments = []
        for c in chunks:
            segs = self.segment_chunk(c["text"])
            for seg in segs:
                seg_id = str(uuid.uuid4())
                chunk_segments.append({
                    "chunk_id": c["chunk_id"],
                    "segment_id": seg_id,
                    "text": seg
                })
                # print(f"chunk {c['chunk_id']} segment {seg_id}:")
                # print(seg)
        self.__chunks = chunks
        self.__chunk_segments = chunk_segments
    
    def get_embedding(self):
        # if self.is_embedding:
        #     return
        self.split()
        
        def add_in_batches(collection, ids, docs, metas, batch=10):
            for i in range(0, len(ids), batch):
                j = i + batch
                collection.add(
                    ids=ids[i:j],
                    documents=docs[i:j],
                    metadatas=metas[i:j],
                )

        # 用法
        add_in_batches(
            self.collection,
            [c["chunk_id"] for c in self.__chunks],
            [c["text"] for c in self.__chunks],
            [{"type":"chunk","source":self.file_name} for _ in self.__chunks],
            batch=10
        )
        add_in_batches(
            self.collection,
            [s["segment_id"] for s in self.__chunk_segments],
            [s["text"] for s in self.__chunk_segments],
            [{"type":"segment","chunk_id":s["chunk_id"],"source":self.file_name} for s in self.__chunk_segments],
            batch=10
        )
        
    def query(self, query_texts:str, n_results=5):
        """按 segment 检索，去重到 chunk 返回。可能少于 n_results。"""
        # 放大检索数量，便于去重
        k = max(10, n_results * 5)

        res = self.collection.query(
            query_texts=query_texts,
            n_results=k,
            where={"type": "segment"},   # 只搜分段
            include=["metadatas", "distances", "documents"]
        )
        # print("res:\n", res)
        
        df = pd.DataFrame({
            "similarity": res["distances"][0],
            "chunk_id": [m["chunk_id"] for m in res["metadatas"][0]]
        })
        # Higher similarity is better.
        df = df.sort_values("similarity", ascending=False).reset_index(drop=True)
        df = df.drop_duplicates(subset="chunk_id", keep="first").reset_index(drop=True)
        # print(df)
        
        duplicate_chunk_ids = df[df["chunk_id"].duplicated()]
        if not duplicate_chunk_ids.empty:
            print("Duplicate chunk_ids found:")
            print(duplicate_chunk_ids)
        else:
            print("No duplicate chunk_ids found.")
            
        
        
        
        # Keep only the top n_results chunks
        top = df.head(n_results).copy()
        chunk_ids = top["chunk_id"].tolist()

        # Fetch the chunk-level documents by chunk_id from the collection
        chunk_records = self.collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"]
        )
        # print("chunk_records\n", chunk_records)
        # Map id -> document and keep the original ranking order
        id_to_doc = {i: d for i, d in zip(chunk_records["ids"], chunk_records["documents"])}
        top["document"] = [id_to_doc[cid] for cid in chunk_ids]

        # print(top)
        return top
            
        
            
    
    
                

if __name__ == "__main__":
    retriever = Retriever("./中华人民共和国刑法_20201226.docx")
    # retriever.split()
    # retriever.get_embedding()
    print("result:\n", retriever.query("把你妈杀了"))
    # data = retriever.collection.get(
    #     where={"type": "segment"},
    #     include=["documents", "metadatas"],   # 如需向量再加 "embeddings"
    #     limit=20, offset=0
    # )
    # print(data["ids"][:2])
    # for i in range(len(data["documents"][:2])):
    #     print(i)
    #     print(data["documents"][i])
    # print(data["metadatas"][:2])
