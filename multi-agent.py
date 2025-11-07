# multi_agent_minimal.py
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# ============== 基础消息与总线 ==============
@dataclass
class Message:
    sender: str
    recipient: str  # 具体名字或 "broadcast"
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)

class MessageBus:
    def __init__(self):
        self._inbox: Dict[str, List[Message]] = {}
        self._broadcast: List[Message] = []
        self._history: List[Message] = []

    def send(self, msg: Message):
        self._history.append(msg)
        if msg.recipient == "broadcast":
            self._broadcast.append(msg)
        else:
            self._inbox.setdefault(msg.recipient, []).append(msg)

    def receive_for(self, agent_name: str) -> List[Message]:
        # 取出专属 + 广播
        direct = self._inbox.pop(agent_name, [])
        msgs = self._broadcast + direct
        self._broadcast = []  # 广播一次性消费（简单做法）
        return msgs

    @property
    def history(self) -> List[Message]:
        return self._history

# ============== Agent 抽象 ==============
class Agent:
    def __init__(self, name: str, bus: MessageBus):
        self.name = name
        self.bus = bus
        self.state: Dict[str, Any] = {}

    async def receive(self, messages: List[Message]):
        """更新内部状态；默认仅记录最近消息。"""
        self.state.setdefault("messages", []).extend(messages)

    async def policy(self) -> Optional[Message]:
        """决定下一条消息。None 表示本轮不发言。"""
        raise NotImplementedError

# ============== 一个可替换的“模型调用” ==============
def llm(prompt: str) -> str:
    """
    这里用规则/模板假装是 LLM，方便无外部依赖时演示。
    你可以替换为真实的 API 调用（例如 OpenAI、DeepSeek 等）。
    """
    if "CRITIC_JUDGE" in prompt:
        # 简单规则：含有 "总结" 或 "完成" 就通过
        if "总结" in prompt or "完成" in prompt:
            return "Looks good. STOP"
        return "Needs revision: 请补充一个简短总结。"
    # 其它情况就复述
    return "草稿回应：" + prompt[:150]

# ============== 三个具体 Agent ==============
class Planner(Agent):
    async def policy(self) -> Optional[Message]:
        if self.state.get("planned"):  # 只规划一次
            return None
        self.state["planned"] = True
        task = self.state.get("task", "请撰写一份关于多智能体的简述")
        subtasks = [
            "解释什么是多智能体与基本组件。",
            "给出一个最小可运行示例。",
            "写一个3行以内的总结。"
        ]
        # 指派给 Researcher
        return Message(
            sender=self.name,
            recipient="Researcher",
            content=f"子任务列表：{subtasks}\n整体任务：{task}",
            meta={"role": "plan"}
        )

class Researcher(Agent):
    async def policy(self) -> Optional[Message]:
        msgs = self.state.get("messages", [])
        # 找到 Planner 的指派
        plan_msgs = [m for m in msgs if m.meta.get("role") == "plan"]
        critic_msgs = [m for m in msgs if m.meta.get("role") == "critique"]

        # 被要求修订
        if critic_msgs:
            need = critic_msgs[-1].content
            draft = self.state.get("draft", "")
            improved = draft + "\n\n补充总结：多智能体=一组自主体通过消息/黑板协作，靠协议和调度达成目标。"
            self.state["draft"] = improved
            return Message(
                sender=self.name,
                recipient="Critic",
                content=f"修订稿：\n{improved}\n（根据需求：{need}）",
                meta={"role": "response"}
            )

        if plan_msgs and not self.state.get("draft"):
            # 首次草拟
            subtasks_text = plan_msgs[-1].content
            draft = (
                "一、概念：多智能体是多个可自治的Agent协作完成任务的范式。\n"
                "核心组件：消息协议、Agent接口、黑板/总线、调度与终止条件。\n\n"
                "二、示例：使用Python + asyncio，自定义MessageBus与Agent抽象。\n"
                "提供Planner/Researcher/Critic三角互动。\n\n"
                "三、总结（待补）："
            )
            self.state["draft"] = draft
            return Message(
                sender=self.name,
                recipient="Critic",
                content=f"初稿：\n{draft}",
                meta={"role": "response"}
            )
        return None

class Critic(Agent):
    async def policy(self) -> Optional[Message]:
        msgs = self.state.get("messages", [])
        # 收到研究稿件就评审
        responses = [m for m in msgs if m.meta.get("role") == "response"]
        if not responses:
            return None

        latest = responses[-1].content
        # 通过“llm”伪判断
        judge = llm("CRITIC_JUDGE\n" + latest)
        if "STOP" in judge:
            return Message(
                sender=self.name,
                recipient="broadcast",
                content="最终通过。任务完成。STOP",
                meta={"role": "stop"}
            )
        else:
            return Message(
                sender=self.name,
                recipient="Researcher",
                content=judge,
                meta={"role": "critique"}
            )

# ============== 调度器 ==============
class Orchestrator:
    def __init__(self, agents: List[Agent], bus: MessageBus, max_rounds: int = 12):
        self.agents = agents
        self.bus = bus
        self.max_rounds = max_rounds

    async def run(self, task: str) -> List[Message]:
        # 初始化：把任务广播给 Planner
        self.bus.send(Message(sender="User", recipient="Planner",
                              content=task, meta={"role": "user_task"}))
        for a in self.agents:
            a.state["task"] = task

        for round_id in range(self.max_rounds):
            # 轮询每个Agent
            for agent in self.agents:
                inbox = self.bus.receive_for(agent.name)
                if inbox:
                    await agent.receive(inbox)
                msg = await agent.policy()
                if msg:
                    self.bus.send(msg)

            # 检查是否 STOP
            for m in self.bus.history[-5:]:
                if m.meta.get("role") == "stop" or "STOP" in m.content:
                    return self.bus.history
            await asyncio.sleep(0)  # 让出事件循环
        return self.bus.history

# ============== 运行示例 ==============
async def main():
    bus = MessageBus()
    planner = Planner("Planner", bus)
    researcher = Researcher("Researcher", bus)
    critic = Critic("Critic", bus)

    orch = Orchestrator([planner, researcher, critic], bus, max_rounds=10)
    history = await orch.run("撰写一段关于“如何不用框架实现多智能体”的简短说明，并给出最小示例。")

    print("=== 消息历史 ===")
    for m in history:
        print(f"[{m.sender} -> {m.recipient}] {m.meta} :: {m.content[:120]}")

if __name__ == "__main__":
    asyncio.run(main())