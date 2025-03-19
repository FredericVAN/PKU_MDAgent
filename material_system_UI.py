import os
import autogen
import asyncio
from autogen import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
)
from autogen.coding import DockerCommandLineCodeExecutor

from utils.llm_json_api import json2dict_from_llm_output
from utils.logger_config import logger

os.environ["OPENAI_API_KEY"] = ""
input_future = None
gpt4o_mini_config_list=[
    {
        "model": "gpt-4o",
        "api_key": os.environ["OPENAI_API_KEY"],

    }
]
llama_31_config_list = [
    {
        # Let's choose the Meta's Llama 3.1 model (model names must match Ollama exactly)
        "model": "llama3.1:8b-instruct-q6_K",
        # We specify the API Type as 'ollama' so it uses the Ollama client class
        "api_type": "ollama",
        "stream": False,
    }
]
#常量
PASS_SCORE = 8
AGNET_DESCRIPTION = {"LammpsWorker":"Expert_in_generating_lammps_script_files",
                     "LammpsEvaluator":"Expert_in_evaluate_lammps_script",
                     "MatlabWorker":"Expert_in_generating_matlab_script",
                     "MatlabEvaluator":"Expert_in_evaluate_matlab_script"}
class MyConversableAgent(autogen.ConversableAgent):

    async def a_get_human_input(self, prompt: str) -> str:
        global input_future
        print('AGET!!!!!!')  # or however you wish to display the prompt
        chat_interface.send(prompt, user="System", respond=False)
        # Create a new Future object for this input operation if none exists
        if input_future is None or input_future.done():
            input_future = asyncio.Future()

        # Wait for the callback to set a result on the future
        await input_future

        # Once the result is set, extract the value and reset the future for the next input operation
        input_value = input_future.result()
        input_future = None
        return input_value

#TODO 1.这里是不用UI的p
# user_proxy = UserProxyAgent(
#     name="Admin",
#     system_message="A human admin. Give the task, and send instructions to writer to refine the blog post.",
#     code_execution_config=False,
# )
user_proxy_with_UI = MyConversableAgent(
    name="Admin",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("exit"),
    system_message="""A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin. 

   """,
    # Only say APPROVED in most cases, and say exit when nothing to be done further. Do not say others.
    code_execution_config=False,
    human_input_mode="ALWAYS",
)
user_proxy = user_proxy_with_UI

planner_system_prompt = f"""
#角色
你将获取一个任务，然后将任务分解为子任务,并为每个子任务分配负责的agent

#目前的可用的agent成员:
{AGNET_DESCRIPTION}

#要求
回答要求非常简洁和规整，不超过200个字。
"""
planner = AssistantAgent(
    name="Planner",
    system_message=planner_system_prompt,
    llm_config={"config_list": gpt4o_mini_config_list, "cache_seed": None},
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
lammps_worker = AssistantAgent(
    name = "Expert_in_generating_lammps_script_files",
    llm_config={"config_list": gpt4o_mini_config_list, "cache_seed": None},
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
lammps_evaluator = AssistantAgent(
    name="Expert_in_evaluate_lammps_script",
    llm_config={"config_list": gpt4o_mini_config_list, "cache_seed": None},
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
matlab_worker = AssistantAgent(
    name="Expert_in_generating_Matlab",
    system_message=matlab_worker_system_prompt,
    llm_config={"config_list": gpt4o_mini_config_list, "cache_seed": None},
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
matlab_evaluator = AssistantAgent(
    name="Expert_in_evaluate_matlab_script",
    system_message=matlab_evaluator_system_prompt,
    llm_config={"config_list": gpt4o_mini_config_list, "cache_seed": None},
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
    llm_config={"config_list": gpt4o_mini_config_list, "cache_seed": None},
    code_execution_config=False,  # Turn off code execution for this agent.
)
from pathlib import Path

# ---jupyter---
# server = DockerJupyterServer()
# executor = JupyterCodeExecutor(server)
workdir = Path("dockerCodeExecutor_env/paper_test")
workdir.mkdir(parents=True, exist_ok=True)
executor = DockerCommandLineCodeExecutor(
    image="python:3.12-slim",  # Execute code using the given docker image name.
    timeout=10,  # Timeout for each code execution in seconds.
    work_dir=workdir.name,  # Use the temporary directory to store the code files.
)
code_executor = UserProxyAgent(
    name="Code_Executor",
    system_message="Executor. Execute the code written by the others and report the result.",
    human_input_mode="ALWAYS",
    code_execution_config={
        "last_n_messages": 3,
        "executor": executor,
    },
)

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

    elif last_speaker is matlab_worker:
        return matlab_evaluator
    elif last_speaker is matlab_evaluator:
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
        if messages[-1]["content"].strip() != "":
            # If the last message is from user and is not empty, let the writer to continue
            return "auto"
    else:
        # default to auto speaker selection method
        return "auto"


groupchat = GroupChat(
    agents=[user_proxy, lammps_worker, lammps_evaluator,matlab_worker,matlab_evaluator,planner],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
)
manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": gpt4o_mini_config_list, "cache_seed": None},name="lammps_matlab_manager")


avatar = {user_proxy.name: "👨‍💼", lammps_worker.name: "👩‍💻", lammps_evaluator.name: "👩‍🔬", planner.name: "🗓", code_executor.name: "🛠",
          code_writer_agent.name: '📝',matlab_worker.name:"‍💻",matlab_evaluator.name:"‍💻"}

import panel as pn
import autogen
def print_messages(recipient, messages, sender, config):
    print(
        f"Messages from: {sender.name} sent to: {recipient.name} | num messages: {len(messages)} | message: {messages[-1]}")

    content = messages[-1]['content']

    if all(key in messages[-1] for key in ['name']):
        chat_interface.send(content, user=messages[-1]['name'], avatar=avatar[messages[-1]['name']], respond=False)
    else:
        chat_interface.send(content, user=recipient.name, avatar=avatar[recipient.name], respond=False)

    return False, None  # required to ensure the agent communication flow continues

user_proxy.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)
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
pn.extension(design="material")
initiate_chat_task_created = False


async def delayed_initiate_chat(agent, recipient, message):
    global initiate_chat_task_created
    # Indicate that the task has been created
    initiate_chat_task_created = True

    # Wait for 2 seconds
    await asyncio.sleep(2)

    # Now initiate the chat
    await agent.a_initiate_chat(recipient, message=message)


async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    global initiate_chat_task_created
    global input_future

    if not initiate_chat_task_created:
        asyncio.create_task(delayed_initiate_chat(user_proxy, manager, contents))

    else:
        if input_future and not input_future.done():
            input_future.set_result(contents)
        else:
            print("There is currently no input being awaited.")

#首先检查os.environ.get('OPENAI_API_KEY')是否有，如果没有就让用户从命令行输入
if not os.environ.get('OPENAI_API_KEY'):
    api_key = input("请输入你的OpenAI API Key: ")
    os.environ['OPENAI_API_KEY'] = api_key
    logger.info("OpenAI API Key: " + api_key)

chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send(f"欢迎使用MaterialAgent,目前针对材料领域的Agent成员有\n:{AGNET_DESCRIPTION}", user="System", respond=False)
chat_interface.servable()
pn.serve(chat_interface, show=True, port=5006)
#TODO 执行运行的话用这个
# with Cache.disk(cache_seed=41) as cache:
#     task_str = f"""
#     1.目标：我想使用Lammps与Matlab两个软件,Calculate the volumetric heat capacity of copper using LAMMPS.
#     2.你的工作: 生成一个Lammps脚本和一个Matlab脚本，让expert对脚本内容进行评估，评估分数均高达9分以上再给我看最终脚本结果。
#     """
#     task_str_2 =f"""
#     Calculate the volumetric heat capacity of copper using LAMMPS.
#     1.生成一个lammps脚本
#     2.让code_executor去保存脚本保存到本地
#     """
#     groupchat_history_custom = user_proxy.initiate_chat(
#         manager,
#         message=task_str_2,
#         cache=cache,
#     )