# -*- coding: utf-8 -*-
# @Author  : Zhuofan Shi
# @Time    : 2024/8/7 17:02
# @File    : llm_common_api.py
# @Software: PyCharm
import json
import os

import httpx
import json_repair

def json2dict_from_llm_output(llm_output_str:str)->dict:
    '''
    从大模型的回答中提取出predict_result
    :param llm_output_str:
    :return:
    '''
    answer_dict={}
    if llm_output_str.startswith("```json"):
        llm_output_str = llm_output_str[7:]
    elif llm_output_str.startswith("```"):
        llm_output_str = llm_output_str[3:]
    if llm_output_str.endswith("```"):
        llm_output_str = llm_output_str[:-3]
    try:
        answer_dict = json_repair.loads(llm_output_str)
    except Exception as e:
        #尝试补救
        llm_output_str.replace("，", ",")
        llm_output_str = llm_output_str.replace("“", "\"")
        llm_output_str = llm_output_str.replace("”", "\"")
        answer_dict = json_repair.loads(llm_output_str)
    return answer_dict

import base64
from mimetypes import guess_type
from openai import OpenAI

# Function to encode a local image into data URL
def get_image_data_from_path(image_path, is_local_img):
    if is_local_img:
        image_data = encode_image_for_openai(image_path)
    else:
        image_data = base64.b64encode(httpx.get(image_path).content).decode("utf-8")
    return image_data
def encode_image_for_openai(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
def local_image_to_data_url_by_base64(image_path):
    '''
    https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest%2Csystem-assigned%2Cresource#use-a-local-image
    :param image_path:
    :return:
    '''
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

if __name__ == "__main__":
    image_path = r'D:\python\evaluate-system\dataset\vsr\images\val2017\000000000139.jpg'
    data_url = local_image_to_data_url_by_base64(image_path)
    print("Data URL:", data_url)
