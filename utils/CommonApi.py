from dotenv import load_dotenv

from langchain_community.chat_models.tongyi import ChatTongyi

import os

from utils.logger_config import logger
# 参考https://python.langchain.com/docs/modules/model_io/llms/custom_llm
from langchain_openai import AzureChatOpenAI
load_dotenv()


def _required_env(name: str) -> str:
    """Return a required setting from .env with a clear error message."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def init_llm(llm_name):
    '''
    Set the language model.
    :param llm_name:
    :return:
    '''
    if llm_name == "Tongyi" or llm_name == "通义千问":
        return ChatTongyi(temperature=0,model='qwen-plus')
    elif llm_name == "qwen-max-longcontext":
        return ChatTongyi(temperature=0,model='qwen-max-longcontext')
    elif llm_name == "qwen-max":
        return ChatTongyi(temperature=0,model='qwen-max')
    elif llm_name == "qwen-plus":
        return ChatTongyi(temperature=0,model='qwen-plus') #普通的超长上下文
    elif llm_name == "qwen-turbo":
        return ChatTongyi(temperature=0,model='qwen-turbo')
    elif llm_name == "gpt-3.5":
        gpt35_model = AzureChatOpenAI(
            azure_endpoint=_required_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_required_env("AZURE_OPENAI_API_KEY"),
            azure_deployment=_required_env("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=_required_env("AZURE_OPENAI_API_VERSION"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )
        return  gpt35_model
    elif llm_name == "gpt-4o":
        gpt4o_model = AzureChatOpenAI(
            azure_endpoint=_required_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_required_env("AZURE_OPENAI_API_KEY"),
            azure_deployment=_required_env(
                "GPT_4O_AZURE_OPENAI_DEPLOYMENT_NAME"
            ),
            api_version=_required_env("GPT_4O_AZURE_OPENAI_API_VERSION"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )
        return  gpt4o_model
    else:
        raise ValueError("Unknown language model: %s" % llm_name)


def create_index_if_not_exists(es_client, index_name):
    '''
    检查如果该索引不存在创建索引
    :param es_url:
    :param index_name:
    :return:
    '''
    #第一步
    if not es_client.indices.exists(index=index_name):
        # 创建索引
        es_client.indices.create(index=index_name)
        logger.debug(f"索引{index_name}不存在，创建索引成功")
        return True
    else:
        logger.debug(f"索引{index_name}已存在")
        return False
if __name__ == '__main__':
    #测试init_hybrid_retriever
    print("启动")
    embedding = init_embeddings("DashScope")
    retriever = init_retriever(embedding=embedding, index_name="knowledge_index", es_url="http://localhost:9200", retriever_name="hybrid_retriever")
    docs = retriever.get_relevant_documents("你好")
    for doc in docs:
        print(doc)
    print(len(docs))
    # print(docs)
