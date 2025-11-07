from docx import Document
import uuid
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
import os
from AiClass import AiClient
import asyncio

api_key = "sk-dfb5d9295ee94adeafc438d18c7d4900"


class Retriever:
    def __init__(
        self,
        file_path,
        db_path="./chroma_db",
        chunk_size=1000,
        overlap=200,
        segment_size=200,
    ):
        """type:summary, partial_summary, segment, chunk"""
        self.file_path = file_path
        self.file_name = file_path.split("/")[-1].split(".")[0]

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.segment_size = segment_size

        self.aiClient = AiClient()

        self.embedding_model = OpenAIEmbeddingFunction(
            model_name="text-embedding-v4",
            api_key=api_key,
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.client = chromadb.PersistentClient(path=db_path)
    
    @classmethod
    async def create(cls, file_path):
        self = cls(file_path)
        # check collection if exists
        collection_names = [c.name for c in self.client.list_collections()]
        self.collection_name = str(uuid.uuid5(uuid.NAMESPACE_DNS, self.file_name))
        
        self.split()
        if self.collection_name in collection_names:
            logging.info(f"Collection {self.collection_name} already exists.")
            self.collection = self.client.get_collection(
                self.collection_name, embedding_function=self.embedding_model
            )
        else:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_model,
                metadata={"hnsw:space": "cosine"},
            )
            self.split()
            await self.get_summary()
            self.get_embedding()
        return self

    def read_docx(self, documents: str):
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

        sentences = re.split(r"(?<=[。！？.!?])", chunk)
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
                chunk_segments.append(
                    {"chunk_id": c["chunk_id"], "segment_id": seg_id, "text": seg}
                )
                # print(f"chunk {c['chunk_id']} segment {seg_id}:")
                # print(seg)
        self.__chunks = chunks
        self.__chunk_segments = chunk_segments
        

    def __add_in_batches(self, collection, ids, docs, metas, batch=10):
        for i in range(0, len(ids), batch):
            j = i + batch
            collection.add(
                ids=ids[i:j],
                documents=docs[i:j],
                metadatas=metas[i:j],
            )
            
    def get_embedding(self):
        # add summary
        self.__add_in_batches(
            self.collection,
            [c["summary_id"] for c in self.__summary],
            [c["text"] for c in self.__summary],
            [
                {"type": "summary", "source": self.file_name}
                for s in self.__summary
            ],
            batch=1,
        )
        
        # add partial summary        
        self.__add_in_batches(
            self.collection,
            [c["par_sum_id"] for c in self.__partial_summaries],
            [c["text"] for c in self.__partial_summaries],
            [
                {"type": "partial_summary", "source": self.file_name}
                for s in self.__partial_summaries
            ],
            batch=10,
        )
        
        # add segment
        self.__add_in_batches(
            self.collection,
            [s["segment_id"] for s in self.__chunk_segments],
            [s["text"] for s in self.__chunk_segments],
            [
                {"type": "segment", "chunk_id": s["chunk_id"], "source": self.file_name}
                for s in self.__chunk_segments
            ],
            batch=10,
        )
        
        # add_chunk
        self.__add_in_batches(
            self.collection,
            [c["chunk_id"] for c in self.__chunks],
            [c["text"] for c in self.__chunks],
            [{"type": "chunk", "par_sum_id": c["par_sum_id"],"source": self.file_name} for c in self.__chunks],
            batch=10,
        )

    async def get_summary(self, size_of_chunks=5, max_concurrency=5):
        # 异步请求 llm
        sem = asyncio.Semaphore(max_concurrency)
        async def call_one(prompt, par_sum_id):
            async with sem:
                resp = await self.aiClient.get_response(prompt)
                return {"par_sum_id": par_sum_id, "text": resp}
            
        self.__partial_summaries = []
        # load summary prompt
        with open("./prompt/summary.txt", "r", encoding="utf-8") as f:
            text = f.read()

        tasks = []
        for i in range(0, len(self.__chunks), size_of_chunks):
            # create id for each partial summary
            par_sum_id = str(uuid.uuid4())
            
            batch = self.__chunks[i:min(i + size_of_chunks, len(self.__chunks))]

            # 标记批次 id（就地修改是否符合你期望）
            for c in batch:
                c["par_sum_id"] = par_sum_id

            texts = [c["text"] for c in batch]
            partial_text = "\n\n".join(texts)
        
            # get prompt
            prompt = text.format(input_text=partial_text)
            prompt = {"messages": [{"role": "user", "content": f"{prompt}"}]}
            # call llm get partial summary and add to partial summaries
            tasks.append(call_one(prompt, par_sum_id))
        
        results = await asyncio.gather(*tasks)
        self.__partial_summaries = results
        
        
        partial_text = "\n\n".join(str(ps["text"]) for ps in self.__partial_summaries)
        # get prompt
        prompt = text.format(input_text=partial_text)
        prompt = {"messages": [{"role": "user", "content": f"{prompt}"}]}
        self.__summary = [{"summary_id":str(uuid.uuid4()), "text": await self.aiClient.get_response(prompt)}]
        

    def query(self, query_texts: str, type="segment", n_results=5):
        """按 segment 检索，去重到 chunk 返回。可能少于 n_results。"""
        if type == "segment":
            # 放大检索数量，便于去重
            k = max(10, n_results * 5)

            res = self.collection.query(
                query_texts=query_texts,
                n_results=k,
                where={"type": type},  # 只搜分段
                include=["metadatas", "distances", "documents"],
            )
            # print("res:\n", res)

            df = pd.DataFrame(
                {
                    "similarity": res["distances"][0],
                    "chunk_id": [m["chunk_id"] for m in res["metadatas"][0]],
                }
            )
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
                ids=chunk_ids, include=["documents", "metadatas"]
            )
            # print("chunk_records\n", chunk_records)
            # Map id -> document and keep the original ranking order
            id_to_doc = {
                i: d for i, d in zip(chunk_records["ids"], chunk_records["documents"])
            }
            top["document"] = [id_to_doc[cid] for cid in chunk_ids]

            # print(top)
            return top
        else:
            res = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where={"type": type},  # 只搜分段
                include=["metadatas", "distances", "documents"],
            )
            res = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where={"type": type},  # 只搜分段
                include=["metadatas", "distances", "documents"],
            )
            # print("res:\n", res)

            df = pd.DataFrame(
                {
                    "similarity": res["distances"][0],
                    "documents": res["documents"][0],
                }
            )
            # Higher similarity is better.
            df = df.sort_values("similarity", ascending=False).reset_index(drop=True)
            return df

    def get_chunk(self):
        self.split()
        return self.__chunks


if __name__ == "__main__":
    retriever = asyncio.run(Retriever.create("./knowledge_files/中华人民共和国刑法_20201226.docx"))

    # retriever.split()
    # retriever.get_embedding()
    df = retriever.query("把你妈杀了",type="summary")
    print(df)
    res = retriever.collection.query(
                query_texts="你好",
                n_results= 1,
                where={"type": "summary"},  # 只搜分段
                include=["metadatas", "distances", "documents"],
            )
    print(res)
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

    # chunks = retriever.get_chunk()
    # print("type chunks:", type(chunks))
    # print("length chunks:", len(chunks))
    # parial_chunks = "\n\n".join(str(c) for c in chunks[:5])
    # print("parial_chunks:", parial_chunks)
    # client = AiClient()
    # with open("./prompt/summary.txt", "r", encoding="utf-8") as f:
    #     text = f.read()
    # prompt = text.format(input_text=parial_chunks)
    # print("prompt:", prompt)
    # prompt = {"messages": [{"role": "user", "content": f"{prompt}"}]}
    # response = client.get_response(prompt)
    # print("-" * 50)
    # print(response)



