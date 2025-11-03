from mcp.server.fastmcp import FastMCP
import datetime
from RetrieverClass import Retriever

# 创建 FastMCP 实例
mcp = FastMCP("DemoMCP")

# 定义一个工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# 定义一个资源
@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """Return a greeting"""
    return f"Hello, {name}!"

# 新增时间工具
@mcp.tool()
def get_current_time() -> str:
    """返回当前系统时间（MCP）"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


import pandas as pd
@mcp.resource("getlaw://{question}")
def get_law_info(question: str) -> str:
    """返回问题相关的中华人民共和国刑法信息"""
    retriever = Retriever("./中华人民共和国刑法_20201226.docx")
    df = retriever.query(question)
    print('type',type(df.to_markdown(index=False)))
    return df.to_markdown(index=False)


if __name__ == "__main__":
    # 启动 stdio 服务器
    mcp.run(transport="stdio")