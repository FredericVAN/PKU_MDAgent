import json
import os

import requests
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

import autogen
from autogen import AssistantAgent, UserProxyAgent, register_function
from autogen.cache import Cache

load_dotenv()

# Import Cognitive Search index ENV
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_API_VERSION = os.getenv("AZURE_SEARCH_API_VERSION")
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG")
AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
credential = DefaultAzureCredential()
endpoint = AZURE_SEARCH_SERVICE_ENDPOINT

from azure.identity import AzureCliCredential

credential = AzureCliCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")

print("TOKEN", token.token)

client = SearchClient(endpoint=endpoint, index_name="test-index", credential=credential)
config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
)

gpt4_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}
def search(query: str):
    payload = json.dumps(
        {
            "search": query,
            "vectorQueries": [{"kind": "text", "text": query, "k": 5, "fields": "vector"}],
            "queryType": "semantic",
            "semanticConfiguration": AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG,
            "captions": "extractive",
            "answers": "extractive|count-3",
            "queryLanguage": "en-US",
        }
    )

    response = list(client.search(payload))

    output = []
    for result in response:
        result.pop("titleVector")
        result.pop("contentVector")
        output.append(result)

    return output

# cog_search = AssistantAgent(
#     name="COGSearch",
#     system_message="You are a helpful AI assistant. "
#     "You can help with Azure Cognitive Search."
#     "Return 'TERMINATE' when the task is done.",
#     llm_config=gpt4_config,
# )
#
# user_proxy = UserProxyAgent(
#     name="User",
#     llm_config=False,
#     is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
#     human_input_mode="NEVER",
# )
#
# register_function(
#     search,
#     caller=cog_search,
#     executor=user_proxy,
#     name="search",
#     description="A tool for searching the Cognitive Search index",
# )