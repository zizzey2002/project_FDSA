import os, sys, logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from RetrieverClass import Retriever
import asyncio

# —— 关闭/静音 chromadb 日志到 stdout ——
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)

BASE = Path(__file__).resolve().parent
LAW_PATH = BASE / "中华人民共和国刑法_20201226.docx"

mcp = FastMCP("DemoMCP")

# 只初始化一次检索器
retriever = asyncio.run(Retriever.create(str(LAW_PATH)))


# # Resource：客户端用 read_resource("getlaw://你的问题") 访问
# @mcp.resource("getlaw://{question}")  # 旧版不支持 content_type
# def get_law_info(question: str) -> str:
#     df = retriever.query(question)
#     # 需要安装 tabulate：pip install tabulate
#     return df.to_markdown(index=False)


# Tool（若你的 FastMCP 版本支持 mcp.tool()，就保留；否则可先注释）
if hasattr(mcp, "tool"):

    @mcp.tool()
    def query_law(question: str, type: str) -> str:
        """
        按问题检索并返回匹配条文（Markdown）

        参数:
        - question: 用户提问。
        - type: 指定检索类型，可选值如下：
            - 'summary'：整篇法律条文总结问答。
            - 'partial_summary'：针对一块知识的综合问答。
            - 'segment'：针对局部知识块的检索问答。

        返回:
        - Markdown 格式的匹配条文。
        """
        df = retriever.query(question, type=type)
        return df.to_markdown(index=False)


if __name__ == "__main__":
    print("MCP server starting...", file=sys.stderr)  # 仅打到 stderr
    # 旧版 FastMCP 常用这个入口
    mcp.run(transport="stdio")
    # 若你环境有 mcp.run_stdio()，也可用：
    # mcp.run_stdio()
