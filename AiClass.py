from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import asyncio

class AiClient:
    def __init__(self,
                OPENROUTER_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                OPENAI_API_KEY = "sk-dfb5d9295ee94adeafc438d18c7d4900", AGENT_MODEL_NAME = "qwen3-max"):
        self.api_key = OPENAI_API_KEY
        self.model_name = AGENT_MODEL_NAME
        self.base_url = OPENROUTER_API_BASE
        self.llm = ChatOpenAI(
            model=AGENT_MODEL_NAME, 
            openai_api_base=OPENROUTER_API_BASE,
            api_key=self.api_key
        )
        self.agent = create_agent(
            self.llm
        )
    
    async def get_response(self, prompt, only_response=True):
        response = await self.agent.ainvoke(prompt)
        if only_response:
            return response['messages'][1].content
        return response

    def get_response(self, prompt, only_response=True):
        response = self.agent.invoke(prompt)
        if only_response:
            return response['messages'][1].content
        return response
    

import os
from openai import OpenAI
class EmbeddingClient:
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-dfb5d9295ee94adeafc438d18c7d4900",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def get_embedding(self, text):
        resp = self.client.embeddings.create(
            model="text-embedding-v4",
            input=[text],
            dimensions=2048
        )
        return resp.data[0].embedding



if __name__ == "__main__":
    
    
    # # test AiClient
    # async def main():
    #     client = AiClient()
    #     prompt = {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    #     response = await client.get_response(prompt)
    #     print(response)
    # asyncio.run(main())
    
    
    # client = AiClient()
    # prompt = {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    # response = client.get_response(prompt)
    # print(response)
    
    client = EmbeddingClient()
    print(client.get_embedding("这是一个测试句子"))