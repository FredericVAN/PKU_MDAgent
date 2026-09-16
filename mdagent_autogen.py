import argparse
import asyncio
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import autogen
import autogen.oai.client as autogen_oai_client
from openai import APIError
from dotenv import load_dotenv
from autogen import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
)
from autogen.coding import DockerCommandLineCodeExecutor
from autogen.io import IOStream

from utils.llm_json_api import json2dict_from_llm_output
from utils.logger_config import logger

load_dotenv()

autogen_provider = os.getenv("AUTOGEN_PROVIDER", "openai").strip().lower()
autogen_model = os.getenv("AUTOGEN_MODEL", "gpt-4o-mini").strip()
autogen_worker_model = os.getenv("AUTOGEN_WORKER_MODEL", autogen_model).strip() or autogen_model
autogen_evaluator_model = os.getenv("AUTOGEN_EVALUATOR_MODEL", autogen_model).strip() or autogen_model
code_executor_mode = os.getenv("AUTOGEN_CODE_EXECUTOR", "disabled").strip().lower()
enable_streaming = os.getenv("AUTOGEN_STREAM", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
stream_retries = max(0, int(os.getenv("AUTOGEN_STREAM_RETRIES", "1")))
stream_fallback = os.getenv("AUTOGEN_STREAM_FALLBACK", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
ollama_think_value = os.getenv("OLLAMA_THINK", "false").strip().lower()
enable_matlab = os.getenv("AUTOGEN_ENABLE_MATLAB", "false").strip().lower() in {
    "1", "true", "yes", "on"
}


_autogen_count_token = autogen_oai_client.count_token


def _count_token_with_fallback(input_data, model: str = "gpt-3.5-turbo-0613") -> int:
    """Keep AutoGen 0.3 streaming usable with newer/custom model names."""
    try:
        return _autogen_count_token(input_data, model=model)
    except NotImplementedError:
        logger.warning(
            "AutoGen cannot count tokens for model %s; using gpt-4o encoding as an estimate.",
            model,
        )
        return _autogen_count_token(input_data, model="gpt-4o")


if enable_streaming:
    autogen_oai_client.count_token = _count_token_with_fallback

if autogen_provider == "openai":
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        openai_api_key = input("请输入你的 OpenAI API Key: ").strip()
        os.environ["OPENAI_API_KEY"] = openai_api_key
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
elif autogen_provider == "ollama":
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    ollama_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
else:
    raise ValueError("AUTOGEN_PROVIDER must be 'openai' or 'ollama'")

request_timeout = float(os.getenv("AUTOGEN_REQUEST_TIMEOUT", "120"))
human_input_handler: Callable[[str], Awaitable[str]] | None = None
message_handler: Callable[[dict[str, str]], None] | None = None
stream_handler: Callable[[dict[str, str]], None] | None = None


def _build_model_config(model: str) -> dict[str, Any]:
    if autogen_provider == "openai":
        config: dict[str, Any] = {
            "model": model,
            "api_key": openai_api_key,
        }
        if openai_base_url:
            config["base_url"] = openai_base_url
        return config

    return {
        "model": model,
        "api_type": "ollama",
        "stream": enable_streaming,
        "client_host": ollama_host,
        "num_predict": ollama_num_predict,
    }


def _build_llm_config(model: str) -> dict[str, Any]:
    return {
        "config_list": [_build_model_config(model)],
        "cache_seed": None,
        "timeout": request_timeout,
        "stream": enable_streaming,
    }


autogen_llm_config = _build_llm_config(autogen_model)
worker_llm_config = _build_llm_config(autogen_worker_model)
evaluator_llm_config = _build_llm_config(autogen_evaluator_model)
PASS_SCORE = int(os.getenv("PASS_SCORE", "8"))
AGNET_DESCRIPTION = {"LammpsWorker":"Expert_in_generating_lammps_script_files",
                     "LammpsEvaluator":"Expert_in_evaluate_lammps_script"}
if enable_matlab:
    AGNET_DESCRIPTION.update({
        "MatlabWorker": "Expert_in_generating_matlab_script",
        "MatlabEvaluator": "Expert_in_evaluate_matlab_script",
    })


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _emit_stream_event(event_type: str, author: str, content: str = "") -> None:
    if stream_handler is not None:
        stream_handler({"type": event_type, "author": author, "content": content})


class _AgentStreamIO:
    """Convert AutoGen's real provider chunks into UI-neutral stream events."""

    def __init__(self, author: str):
        self.author = author

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        raw_text = sep.join(str(item) for item in objects) + end
        text = _ANSI_ESCAPE.sub("", raw_text)
        # AutoGen surrounds streamed completions with ANSI color/reset markers.
        if "\x1b" in raw_text and not text.strip():
            return
        if text:
            _emit_stream_event("token", self.author, text)

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        return input(prompt)


def _parse_ollama_think() -> bool | str | None:
    if ollama_think_value in {"", "auto", "none"}:
        return None
    if ollama_think_value in {"1", "true", "yes", "on"}:
        return True
    if ollama_think_value in {"0", "false", "no", "off"}:
        return False
    if ollama_think_value in {"low", "medium", "high"}:
        return ollama_think_value
    raise ValueError("OLLAMA_THINK must be true, false, auto, low, medium, or high")


def _install_ollama_stream_bridge() -> None:
    """Expose chunks buffered internally by AutoGen 0.3's Ollama adapter."""
    import autogen.oai.ollama as autogen_ollama

    original_client = autogen_ollama.Client
    if getattr(original_client, "_mdagent_stream_bridge", False):
        return

    think = _parse_ollama_think()

    class StreamingOllamaClient(original_client):
        _mdagent_stream_bridge = True

        def chat(self, *args, **kwargs):
            if think is not None:
                kwargs.setdefault("think", think)
            response = super().chat(*args, **kwargs)
            if not kwargs.get("stream", False):
                return response

            def iter_chunks():
                for chunk in response:
                    message = chunk["message"]
                    content = message["content"] or ""
                    if content:
                        IOStream.get_default().print(content, end="", flush=True)
                    yield chunk

            return iter_chunks()

    autogen_ollama.Client = StreamingOllamaClient


if autogen_provider == "ollama" and enable_streaming:
    _install_ollama_stream_bridge()


class StreamingAssistantAgent(AssistantAgent):
    """AssistantAgent that exposes AutoGen 0.3 provider chunks to the UI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # AutoGen 0.3 registers the base coroutine as a function object during
        # initialization, so a normal subclass override is otherwise skipped
        # by a_generate_reply()/GroupChat.
        self.replace_reply_func(
            ConversableAgent.a_generate_oai_reply,
            StreamingAssistantAgent.a_generate_oai_reply,
        )

    async def a_generate_oai_reply(self, messages=None, sender=None, config=None):
        if not enable_streaming or stream_handler is None:
            return await super().a_generate_oai_reply(messages, sender, config)

        for attempt in range(stream_retries + 1):
            _emit_stream_event("start", self.name)
            try:
                with IOStream.set_default(_AgentStreamIO(self.name)):
                    reply = await super().a_generate_oai_reply(messages, sender, config)
            except (APIError, UnboundLocalError) as exc:
                # Some OpenAI-compatible providers return HTTP 200 and then an
                # error event inside the SSE stream. Discard that partial UI
                # message before retrying so users never see duplicated text.
                _emit_stream_event("abort", self.name)
                if attempt < stream_retries:
                    logger.warning(
                        "Streaming response for %s failed (%s); retrying %d/%d.",
                        self.name,
                        exc,
                        attempt + 1,
                        stream_retries,
                    )
                    continue
                if stream_fallback:
                    logger.warning(
                        "Streaming response for %s still failed; falling back to "
                        "one non-streaming request.",
                        self.name,
                    )
                    return await self._a_generate_non_streaming_reply(
                        messages,
                        sender,
                        config,
                    )
                raise
            else:
                _emit_stream_event("end", self.name)
                return reply

        raise RuntimeError("unreachable streaming retry state")

    async def _a_generate_non_streaming_reply(self, messages, sender, config):
        """Temporarily disable stream on this wrapper for a recovery request."""
        client_configs = getattr(self.client, "_config_list", [])
        missing = object()
        original_values = [item.get("stream", missing) for item in client_configs]
        try:
            for item in client_configs:
                item["stream"] = False
            return await super().a_generate_oai_reply(messages, sender, config)
        finally:
            for item, original in zip(client_configs, original_values):
                if original is missing:
                    item.pop("stream", None)
                else:
                    item["stream"] = original


class InteractiveUserProxyAgent(autogen.ConversableAgent):

    async def a_get_human_input(self, prompt: str) -> str:
        if human_input_handler is not None:
            return await human_input_handler(prompt)
        return await asyncio.to_thread(input, prompt)

user_proxy = InteractiveUserProxyAgent(
    name="Admin",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("exit"),
    system_message="""A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.

   """,
    # Only say APPROVED in most cases, and say exit when nothing to be done further. Do not say others.
    code_execution_config=False,
    human_input_mode="ALWAYS",
)

planner_system_prompt = f"""
#角色
你将获取一个任务，然后将任务分解为子任务,并为每个子任务分配负责的agent

#目前的可用的agent成员:
{AGNET_DESCRIPTION}

#要求
回答要求非常简洁和规整，不超过200个字。
"""
planner = StreamingAssistantAgent(
    name="Planner",
    system_message=planner_system_prompt,
    llm_config=autogen_llm_config,
)


lammps_worker_system_prompt = f"""
# 角色
你是一位顶尖的材料领域专家，拥有广博的材料科学知识和丰富的实践经验，尤其精通运用 lammps 进行材料模拟与分析，能够迅速且出色地完成各类材料相关任务。

## 技能
### 技能 1：根据需求提供 lammps 脚本内容
1. 对于需要输入的脚本文件，根据用户需求直接生成完整的脚本内容，并通过注释解释脚本的作用。

## 限制
- 所输出的内容必须按照给定的格式进行组织，不能偏离框架要求。
- 一次性给出完整的Lammps脚本内容
- 脚本文件的解释要清晰明了，易于理解。
"""
lammps_worker_system_prompt += """

If essential information is missing and no safe engineering default can be
used, reply with exactly `NEEDS_USER_INPUT: <one concise question>` and do not
generate a script yet. Otherwise state any reasonable defaults and continue.
Do not expose analysis, drafts, uncertainty monologues, or self-corrections.
Return exactly one final, runnable LAMMPS script in one fenced code block,
followed by at most five short explanatory bullets. Never output invalid
alternatives, placeholders, or commands that you have already rejected.
"""
lammps_worker = StreamingAssistantAgent(
    name = "Expert_in_generating_lammps_script_files",
    llm_config=worker_llm_config,
    system_message=lammps_worker_system_prompt
)
lammps_evaluator_system_prompt = """
# 角色
你是一位极具权威的材料领域专家，具备深厚的材料科学知识与丰富的实践经验。
能够十分严厉严格且准确评估 lammps 脚本内容对用户任务的完成情况，从而以JSON格式给出整数"score"及"reason"。

## 技能
### 技能 1：评估 lammps 脚本
1. 仔细分析 lammps 脚本内容，判断其是否能有效完成用户任务。
2. 严格按照扣分制度，从满分 10 分开始，发现一处错误则扣除对应分数。必须严格要求，不要放过任何一处错误。
3. 以 JSON 格式给出评分结果，仅包含"score"和"reason"。

##参考的扣分规则
语法错误：使用了不存在的命令或拼写错误。扣1分。
简单逻辑错误：例如结构类型错误。扣1分。
参数错误：例如参数设置不正确。扣1分。
关键逻辑错误：例如错误的计算方法。扣2分。
缺少逻辑：例如未设置必要的函数或控制。扣2分。
...(省略其他)

## 限制
- 仅评估 lammps 脚本与材料领域相关的内容。
- 评分和理由必须客观、专业且准确。
- 严格按照 JSON 格式输出结果。
- 除了JSON格式的输出以外,不能再输出其他任务内容。
"""
lammps_evaluator = StreamingAssistantAgent(
    name="Expert_in_evaluate_lammps_script",
    llm_config=evaluator_llm_config,
    system_message=lammps_evaluator_system_prompt
)

matlab_worker_system_prompt = f"""
# 角色
你是一位精通 Matlab 且对材料领域有深入了解的专家。

## 技能
### 技能 1：材料数据分析
1. 当用户提供材料领域的数据时，使用 Matlab 进行数据分析，包括但不限于统计分析、曲线拟合等。
2. 解释分析结果，说明数据所反映的材料特性。

### 技能 2：材料模拟
1. 使用 Matlab 进行材料模拟，如晶体结构模拟、力学性能模拟等。
2. 展示模拟结果，并解释模拟结果对材料性能的影响。

## 限制
- 只回答与 Matlab 在材料领域的应用相关的问题。
- 所输出的内容必须按照给定的格式进行组织，不能偏离框架要求。
"""
matlab_worker_system_prompt += """

If essential information is missing and no safe engineering default can be
used, reply with exactly `NEEDS_USER_INPUT: <one concise question>` and do not
generate a script yet. Otherwise state any reasonable defaults and continue.
"""
matlab_worker = StreamingAssistantAgent(
    name="Expert_in_generating_Matlab",
    system_message=matlab_worker_system_prompt,
    llm_config=worker_llm_config,
)
matlab_evaluator_system_prompt = """
# 角色
你是一位极具权威的材料领域专家，具备深厚的材料科学知识与丰富的实践经验。
能够准确评估 matlab 脚本内容对用户任务的完成情况，从而以JSON格式给出"score"及"reason"。

## 技能
### 技能 1：评估 matlab 脚本
1. 仔细分析 matlab 脚本内容，判断其是否能有效完成用户任务。
2. 严格按照扣分制度，从满分 10 分开始，发现一处错误则扣除一分。不要放过任何一处错误。
3. 以 JSON 格式给出评分结果，仅包含"score"和"reason"。

## 限制
- 仅评估 matlab 脚本与材料领域相关的内容。
- 评分和理由必须客观、专业且准确。
- 严格按照 JSON 格式输出结果。
- 除了JSON格式的输出以外,不能再输出其他任务内容。
"""
matlab_evaluator = StreamingAssistantAgent(
    name="Expert_in_evaluate_matlab_script",
    system_message=matlab_evaluator_system_prompt,
    llm_config=evaluator_llm_config,
)


code_writer_system_message = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply 'TERMINATE' in the end when everything is done.
"""
code_writer_agent = ConversableAgent(
    "code_writer_agent",
    system_message=code_writer_system_message,
    llm_config=autogen_llm_config,
    code_execution_config=False,  # Turn off code execution for this agent.
)
# ---jupyter---
# server = DockerJupyterServer()
# executor = JupyterCodeExecutor(server)
if code_executor_mode == "docker":
    workdir = Path("dockerCodeExecutor_env/paper_test")
    workdir.mkdir(parents=True, exist_ok=True)
    executor = DockerCommandLineCodeExecutor(
        image="python:3.12-slim",  # Execute code using the given docker image name.
        timeout=10,
        work_dir=workdir.name,  # Use the temporary directory to store the code files.
    )
    code_execution_config = {"last_n_messages": 3, "executor": executor}
elif code_executor_mode == "disabled":
    code_execution_config = False
else:
    raise ValueError("AUTOGEN_CODE_EXECUTOR must be 'disabled' or 'docker'")

code_executor = UserProxyAgent(
    name="Code_Executor",
    system_message="Executor. Execute the code written by the others and report the result.",
    human_input_mode="ALWAYS",
    code_execution_config=code_execution_config,
)


def _needs_user_input(message: dict[str, Any]) -> bool:
    return str(message.get("content", "")).lstrip().startswith("NEEDS_USER_INPUT:")


def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat):
    """Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Returns:
        Return an `Agent` class or a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
    """
    messages = groupchat.messages

    if len(messages) <= 1:
        # first, let the engineer retrieve relevant data
        return planner

    if last_speaker is planner:
        # if the last message is from planner, let the engineer to write code
        return lammps_worker

    elif last_speaker is lammps_worker and _needs_user_input(messages[-1]):
        return user_proxy

    elif last_speaker is lammps_worker:
        return lammps_evaluator

    elif last_speaker is lammps_evaluator:
        #解析content
        content = messages[-1]["content"]
        try:
            content_dict = json2dict_from_llm_output(content)
            score = content_dict.get("score", 0)
            if score >= PASS_SCORE:
                return user_proxy
            else:
                return lammps_worker
        except:
            return "auto"

    elif enable_matlab and last_speaker is matlab_worker and _needs_user_input(messages[-1]):
        return user_proxy
    elif enable_matlab and last_speaker is matlab_worker:
        return matlab_evaluator
    elif enable_matlab and last_speaker is matlab_evaluator:
        content = messages[-1]["content"]
        try:
            content_dict = json2dict_from_llm_output(content)
            score = content_dict.get("score", 0)
            if score >= PASS_SCORE:
                return user_proxy
            else:
                return matlab_worker
        except:
            return "auto"

    elif last_speaker is user_proxy:
        if len(messages) >= 2 and _needs_user_input(messages[-2]):
            requesting_agent = messages[-2].get("name")
            if requesting_agent == lammps_worker.name:
                return lammps_worker
            if enable_matlab and requesting_agent == matlab_worker.name:
                return matlab_worker
        if messages[-1]["content"].strip() != "":
            # If the last message is from user and is not empty, let the writer to continue
            return "auto"
    else:
        # default to auto speaker selection method
        return "auto"


groupchat_agents = [user_proxy, lammps_worker, lammps_evaluator, planner]
if enable_matlab:
    groupchat_agents.extend([matlab_worker, matlab_evaluator])

groupchat = GroupChat(
    agents=groupchat_agents,
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
)
manager_name = "lammps_matlab_manager" if enable_matlab else "lammps_manager"
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=autogen_llm_config,
    name=manager_name,
)


def print_messages(recipient, messages, sender, config):
    logger.info(
        "Message from %s to %s | message_count=%d | message=%s",
        sender.name,
        recipient.name,
        len(messages),
        messages[-1],
    )

    message = messages[-1]
    content = str(message.get("content", ""))
    sender_name = str(message.get("name") or sender.name)
    if message_handler is not None:
        message_handler({"type": "message", "author": sender_name, "content": content})

    return False, None  # required to ensure the agent communication flow continues

user_proxy.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)
if enable_matlab:
    matlab_worker.register_reply(
        [autogen.Agent, None],
        reply_func=print_messages,
        config={"callback": None},
    )
    matlab_evaluator.register_reply(
        [autogen.Agent, None],
        reply_func=print_messages,
        config={"callback": None},
    )
lammps_worker.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)
lammps_evaluator.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)
code_writer_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)
code_executor.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

planner.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)


def set_human_input_handler(
    handler: Callable[[str], Awaitable[str]] | None,
) -> None:
    """Install the UI-specific callback used for human-in-the-loop prompts."""
    global human_input_handler
    human_input_handler = handler


def set_message_handler(
    handler: Callable[[dict[str, str]], None] | None,
) -> None:
    """Install a callback that receives messages emitted by the agents."""
    global message_handler
    message_handler = handler


def set_stream_handler(
    handler: Callable[[dict[str, str]], None] | None,
) -> None:
    """Install a callback for real model start/token/end events."""
    global stream_handler
    stream_handler = handler


def reset_conversation() -> None:
    """Clear group and per-agent histories before starting an independent task."""
    groupchat.messages.clear()
    for agent in [*groupchat_agents, manager]:
        agent.clear_history()


async def run_chat_async(task: str):
    """Run one AutoGen workflow without binding the core to a particular UI."""
    initial_task = task.strip()
    if not initial_task:
        raise ValueError("任务不能为空。")
    reset_conversation()
    logger.info(
        "Starting async chat with manager=%s, planner_model=%s, worker_model=%s, evaluator_model=%s, request_timeout=%ss, streaming=%s",
        manager.name,
        autogen_model,
        autogen_worker_model,
        autogen_evaluator_model,
        request_timeout,
        enable_streaming,
    )
    return await user_proxy.a_initiate_chat(
        manager,
        message=initial_task,
        clear_history=False,
    )


def run_cli(task: str | None = None) -> None:
    initial_task = task or input("请输入材料模拟任务：").strip()
    if not initial_task:
        raise ValueError("CLI 模式需要提供非空任务。")
    reset_conversation()
    logger.info(
        "Starting CLI chat with manager=%s, planner_model=%s, worker_model=%s, evaluator_model=%s, request_timeout=%ss, streaming=%s",
        manager.name,
        autogen_model,
        autogen_worker_model,
        autogen_evaluator_model,
        request_timeout,
        enable_streaming,
    )
    try:
        user_proxy.initiate_chat(manager, message=initial_task)
    except Exception:
        logger.exception("CLI chat failed")
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MaterialAgent in terminal mode.")
    parser.add_argument(
        "--mode",
        choices=("cli",),
        default="cli",
        help="Retained for compatibility; the web UI now uses mdagent_chainlit.py.",
    )
    parser.add_argument(
        "--task",
        help="Initial task. If omitted, the terminal prompts for it.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    run_cli(args.task)


if __name__ == "__main__":
    main()
