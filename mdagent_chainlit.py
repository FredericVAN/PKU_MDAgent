"""Chainlit web UI for the legacy AutoGen MaterialAgent workflow."""

import asyncio
import os
from contextlib import suppress

import chainlit as cl

import mdagent_autogen as agent_core


ASK_TIMEOUT = int(os.getenv("CHAINLIT_ASK_TIMEOUT", "600"))
_workflow_lock = asyncio.Lock()


async def _ask_user(prompt: str) -> str:
    """Pause AutoGen and collect approval or clarification in the same chat."""
    response = await cl.AskUserMessage(
        content=prompt or "请确认是否继续，或补充所需信息。",
        timeout=ASK_TIMEOUT,
    ).send()
    if not response:
        raise TimeoutError("等待用户输入超时。")

    answer = str(response.get("output", "")).strip()
    if not answer:
        raise ValueError("用户输入不能为空。")
    return answer


async def _publish_agent_messages(
    workflow_task: asyncio.Task,
    message_queue: asyncio.Queue[dict[str, str]],
) -> None:
    """Forward full messages and real provider tokens to Chainlit."""
    active_streams: dict[str, cl.Message] = {}
    completed_streams: list[tuple[str, str]] = []

    while not workflow_task.done() or not message_queue.empty():
        try:
            event = await asyncio.wait_for(message_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue

        event_type = event.get("type", "message")
        author = event["author"]
        content = event["content"]

        if event_type == "start":
            active_streams[author] = cl.Message(content="", author=author)
            continue

        if event_type == "token":
            stream_message = active_streams.setdefault(
                author,
                cl.Message(content="", author=author),
            )
            await stream_message.stream_token(content)
            continue

        if event_type == "end":
            stream_message = active_streams.pop(author, None)
            if stream_message is not None and stream_message.content:
                await stream_message.send()
                completed_streams.append((author, str(stream_message.content)))
            continue

        if event_type == "abort":
            stream_message = active_streams.pop(author, None)
            if stream_message is not None and stream_message.content:
                await stream_message.remove()
            continue

        # Chainlit already renders messages typed by the human.
        if author == agent_core.user_proxy.name:
            continue

        signature = (author, content)
        if signature in completed_streams:
            completed_streams.remove(signature)
            continue
        await cl.Message(content=content, author=author).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    enabled_agents = "、".join(agent_core.AGNET_DESCRIPTION)
    await cl.Message(
        content=(
            "欢迎使用 MaterialAgent（AutoGen）。\n\n"
            f"当前启用的专业角色：{enabled_agents}。\n\n"
            "直接输入材料模拟任务即可；当 Agent 需要参数、澄清或审批时，"
            "界面会暂停并等待你的回答。"
        ),
        author="System",
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    if _workflow_lock.locked():
        await cl.Message(
            content="当前已有一个 AutoGen 工作流在运行，请等待它结束。",
            author="System",
        ).send()
        return

    async with _workflow_lock:
        message_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        event_loop = asyncio.get_running_loop()

        def enqueue_event(event: dict[str, str]) -> None:
            event_loop.call_soon_threadsafe(message_queue.put_nowait, event)

        agent_core.set_human_input_handler(_ask_user)
        agent_core.set_message_handler(enqueue_event)
        agent_core.set_stream_handler(enqueue_event)

        workflow_task = asyncio.create_task(agent_core.run_chat_async(message.content))
        cl.user_session.set("workflow_task", workflow_task)
        try:
            await _publish_agent_messages(workflow_task, message_queue)
            await workflow_task
        except asyncio.CancelledError:
            await cl.Message(content="工作流已停止。", author="System").send()
        except Exception as exc:
            agent_core.logger.exception("Chainlit chat failed")
            await cl.Message(
                content=f"Agent 请求失败：{exc}\n\n详细堆栈已写入 `myrun.log`。",
                author="System",
            ).send()
        finally:
            agent_core.set_human_input_handler(None)
            agent_core.set_message_handler(None)
            agent_core.set_stream_handler(None)
            cl.user_session.set("workflow_task", None)


@cl.on_stop
async def on_stop() -> None:
    workflow_task = cl.user_session.get("workflow_task")
    if workflow_task and not workflow_task.done():
        workflow_task.cancel()
        with suppress(asyncio.CancelledError):
            await workflow_task
