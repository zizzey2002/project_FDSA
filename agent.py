# run_agent_no_langchain.py
import os, sys, json, asyncio
from typing import Dict, Any

# === MCP 客户端 ===
from mcp import ClientSession
from mcp.client.stdio import stdio_client  # 若该模块存在
from mcp import StdioServerParameters 

# === OpenAI 兼容 SDK（可指向 DashScope compatible-mode）===
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
SERVER = [sys.executable, os.path.join(BASE, "mcp_server.py")]   # 你的 FastMCP 服务

# 你的大模型（支持函数调用）
MODEL = os.getenv("LLM_MODEL", "qwen3-max")  # 也可改成 gpt-4o-mini 等
client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),            # 若用 OpenAI，换成 OPENAI_API_KEY
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

def mcp_tools_to_openai(tools_result) -> list:
    """Convert MCP tools into OpenAI tools schema.
    Accepts either a plain iterable of tools, or the result object from list_tools()
    which may have a .tools attribute.
    Handles tuple, dict, or object-style tool descriptors.
    """
    def _get(obj, *keys, default=None):
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                return obj[k]
            if hasattr(obj, k):
                return getattr(obj, k)
        return default

    def _normalize_tool(t):
        # Common tuple shape: (name, description, schema)
        if isinstance(t, tuple):
            name = t[0] if len(t) > 0 else None
            description = t[1] if len(t) > 1 and isinstance(t[1], str) else ""
            schema = t[2] if len(t) > 2 else {"type": "object", "properties": {}}
            return name, description, schema

        # Dict or object with attributes
        name = _get(t, "name", "tool_name")
        description = _get(t, "description", default="") or ""
        schema = (
            _get(t, "input_schema", "inputSchema", "parameters", "schema")
            or {"type": "object", "properties": {}}
        )
        return name, description, schema

    # Unwrap list_tools() result if needed
    tools_list = getattr(tools_result, "tools", tools_result)

    out = []
    for t in tools_list or []:
        name, description, schema = _normalize_tool(t)
        if not name:
            continue
        if not isinstance(schema, dict) or "type" not in schema:
            schema = {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema
            }
        })

    # Add a generic "read_resource" function
    out.append({
        "type": "function",
        "function": {
            "name": "read_resource",
            "description": "Read an MCP resource by URI, e.g. greeting://Alice or getlaw://杀人罪",
            "parameters": {
                "type": "object",
                "properties": {"uri": {"type": "string"}},
                "required": ["uri"]
            }
        }
    })
    return out

async def call_mcp_tool(session: ClientSession, name: str, args: Dict[str, Any]) -> str:
    """把模型的函数调用转发到 MCP 工具/资源，返回纯文本"""
    if name == "read_resource":
        resp = await session.read_resource(args["uri"])
        contents = getattr(resp, "contents", None) or resp
    else:
        resp = await session.call_tool(name, args)
        contents = getattr(resp, "content", None) or getattr(resp, "contents", None) or resp

    # 取第一段文本；否则转成 JSON
    if isinstance(contents, list) and contents:
        part = contents[0]
        txt = getattr(part, "text", None)
        if txt is not None:
            return txt
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return part["text"]
    try:
        return json.dumps(contents, ensure_ascii=False)
    except Exception:
        return str(contents)

async def chat_with_mcp(user_prompt: str):
    params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(BASE, "mcp_server.py")],
        cwd=BASE,
    )
    print("Starting MCP server:", SERVER)
    # 使用 stdio_client 创建读写流
    async with stdio_client(params) as (read_stream, write_stream):
        # 用两个流初始化 ClientSession
        async with ClientSession(read_stream, write_stream) as mcp_sess:
            await mcp_sess.initialize()
            # 以下逻辑保持不变……
            mcp_tools = await mcp_sess.list_tools()
            tools_for_model = mcp_tools_to_openai(mcp_tools)

            messages = [{"role": "user", "content": user_prompt}]
            while True:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools_for_model,
                    tool_choice="auto",
                )
                msg = resp.choices[0].message
                tool_calls = msg.tool_calls or []

                # Append the assistant message that triggered tool_calls
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or None,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

                if not tool_calls:
                    print(msg.content)
                    break

                for tc in tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    result_text = await call_mcp_tool(mcp_sess, name, args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": result_text
                    })

if __name__ == "__main__":
    # 示例：让模型用 MCP 的法律检索资源与工具
    prompt = "根据中华人民共和国刑法，嫖娼怎么样"
    asyncio.run(chat_with_mcp(prompt))