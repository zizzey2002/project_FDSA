# run_agent.py
import asyncio, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

BASE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(BASE, "mcp_server.py")



from pydantic import BaseModel
from langchain_core.tools import StructuredTool

from pydantic import BaseModel
from langchain_core.tools import StructuredTool
import json
from typing import Any
async def make_read_resource_tool(session):
    class ReadReq(BaseModel):
        uri: Any

    async def _read(uri: str):
        resp = await session.read_resource(uri)

        # 统一拿到 contents 列表：兼容对象 / (uri, contents) / 直接 contents
        if hasattr(resp, "contents"):
            contents = resp.contents
        elif isinstance(resp, tuple) and len(resp) == 2:
            contents = resp[1]
        else:
            contents = resp

        # 从 contents 提取首段纯文本；否则返回 JSON
        for item in contents or []:
            txt = getattr(item, "text", None)
            if isinstance(txt, str):
                return txt
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                return item["text"]

        try:
            return json.dumps(contents, ensure_ascii=False)
        except Exception:
            return str(contents)

    return StructuredTool.from_function(
        coroutine=_read,
        name="mcp_read_resource",
        description="Read an MCP resource by URI, e.g. greeting://Alice or getlaw://杀人罪",
        args_schema=ReadReq,
    )
    
    
async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", SERVER],
        cwd=BASE,
    )
    # 连接 MCP 服务器（stdio）
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            # 把 MCP 工具转换成 LangChain 工具
            tools = await load_mcp_tools(session)
            tools.append(await make_read_resource_tool(session))  # 新增“读资源”工具
            # 创建 LangGraph 智能体（OpenAI 模型示例）
            llm = ChatOpenAI(
                model="qwen3-max",
                openai_api_key=os.getenv("ALIYUN_API_KEY"),
                openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            agent = create_react_agent(llm, tools)

            # 让 LLM 通过 MCP 工具求值
            out = await agent.ainvoke({"messages": "根据中华人民共和国刑法，贪腐判多少年"})
            print(out)



if __name__ == "__main__":
    asyncio.run(main())