import asyncio
from pathlib import Path

from AutoGen_agentchat.agents import AssistantAgent, CodeExecutorAgent, UserProxyAgent
from AutoGen_agentchat.conditions import TextMentionTermination
from AutoGen_agentchat.teams import RoundRobinGroupChat
from AutoGen_agentchat.ui import Console
from AutoGen_ext.code_executors.local import LocalCommandLineCodeExecutor
from AutoGen_ext.models.openai import OpenAIChatCompletionClient

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4.1-mini")

    coder = AssistantAgent(
        "coder",
        model_client=model_client,
        system_message=(
            "You are a senior engineer. Think step-by-step, then output ONLY runnable "
            "Python inside ```pythonthon``` blocks—no commentary."
        ),
    )

    executor = CodeExecutorAgent(
        "executor",
        model_client=model_client,  # lets it narrate results
        code_executor=LocalCommandLineCodeExecutor(work_dir=Path.cwd() / "runs"),
    )

    user = UserProxyAgent("user")  # human in the loop

    termination = TextMentionTermination("exit", sources=["user"])
    team = RoundRobinGroupChat(
        [user, coder, executor], termination_condition=termination
    )

    try:
        await Console(
            team.run_stream()
        )
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())