import os, sys, json, asyncio
from typing import Dict, Any
import re
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
SERVER = [sys.executable, os.path.join(BASE, "mcp_server.py")]  # mcp file path

# build llm client
MODEL = os.getenv("LLM_MODEL", "qwen-turbo")  # model name
client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"), # api_key
    base_url=os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
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
        schema = _get(t, "input_schema", "inputSchema", "parameters", "schema") or {
            "type": "object",
            "properties": {},
        }
        return name, description, schema

    # Unwrap list_tools() result if needed
    tools_list = getattr(tools_result, "tools", tools_result)
    print("-" * 50)
    print("tools_list:", tools_list)
    print("-" * 50)
    out = []
    for t in tools_list or []:
        name, description, schema = _normalize_tool(t)
        if not name:
            continue
        if not isinstance(schema, dict) or "type" not in schema:
            schema = {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            }
        )
        # if not out:
        #     # Add a generic "read_resource" function
        #     out.append(
        #         {
        #             "type": "function",
        #             "function": {
        #                 "name": "read_resource",
        #                 "description": "Read an MCP resource by URI, e.g. or getlaw://杀人罪",
        #                 "parameters": {
        #                     "type": "object",
        #                     "properties": {"uri": {"type": "string"}},
        #                     "required": ["uri"],
        #                 },
        #             },
        #         }
        #     )
        print("-" * 50)
        print("MCP tools:", out)
        print("-" * 50)
    return out


async def call_mcp_tool(session: ClientSession, name: str, args: Dict[str, Any]) -> str:
    """把模型的函数调用转发到 MCP 工具/资源，返回纯文本"""
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

                print("-" * 50)
                print("messages:\n", re.sub(r"\s+", "", str(messages)))
                print("-" * 50)

                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools_for_model,
                    tool_choice="auto",
                )
                print("-" * 50)
                print("MCP response:", resp)
                print("-" * 50)
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

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": result_text,
                        }
                    )


if __name__ == "__main__":
    # 示例：让模型用 MCP 的法律检索资源与工具
    prompt = "根据中华人民共和国刑法，主要讲了什么"
    asyncio.run(chat_with_mcp(prompt))
