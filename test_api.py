import os
import warnings
import traceback
import threading, re
import sys
import time

import requests
import base64
import json
from openai import OpenAI

USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_start|>assistant\n"
IM_END = "<|im_end|>\n"
IMG_TAG = "<img></img>\n"
AUDIO_TAG = "<audio></audio>\n"


def handle_url(url):
    if url.startswith("file://"):
        with open(url[7:], "rb") as f:
            return base64.b64encode(f.read()).decode()
    else:
        req = requests.get(url)
        return base64.b64encode(req.content).decode()


def api_request(
        url, messages,
        stream=False,
        temperature=0.0,
        top_k=20,
        top_p=0.25,
        repetition_penalty=1.05,
        max_new_tokens=4096,
        do_sample=False,
        timeout=3600,
        timeout_stream=180):
    query = ""
    if messages[0]["role"] == "system":
        if messages[0]["content"] != "":
            query += "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
        messages = messages[1:]
    images = []

    for message in messages:
        if message["role"] == "user":
            query += USER_START
            if isinstance(message["content"], list):
                query_content = ""
                img_cnt = 0
                for content in message["content"]:
                    if content["type"] == "image_url":
                        query += IMG_TAG
                        img_cnt += 1
                        images.append(
                            {"type": "base64", "data": handle_url(content["image_url"])}
                        )
                    elif content["type"] == "text":
                        query_content = content["text"]
                    else:
                        raise ValueError("type must be text, image_url")
                if img_cnt >= 2:
                    query += f"用户本轮上传了{img_cnt}张图\n"
                query += query_content + IM_END
            else:
                query += message["content"] + IM_END
        elif message["role"] == "assistant":
            query += ASSISTANT_START
            query += message["content"] + IM_END
        else:
            raise ValueError("role must be user or assistant")
    query += ASSISTANT_START

    play_load = {
        "inputs": query,
        "parameters": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "stop_sequences": ["<|im_end|>"],
            # "skip_special_tokens": False,
        },
    }
    multimodal_params = {}

    if images:
        multimodal_params["images"] = images
    if multimodal_params:
        play_load["multimodal_params"] = multimodal_params

    headers = {"Content-Type": "application/json"}

    if not stream:
        # 非流式
        response = requests.post(url, headers=headers, data=json.dumps(play_load), timeout=timeout)
        response.raise_for_status()
        response = response.json()
        if isinstance(response, list):
            return response[0]["generated_text"]
        return response["generated_text"][0]
    
    else:
        # 流式
        response_iter = requests.post(
            # "http://0.0.0.0:8080/generate_stream", # 服务器IP地址
            url,
            headers=headers,
            data=json.dumps(play_load),
            stream=True,
            timeout=timeout_stream,
        )
        gen_text = ""
        for chunk in response_iter.iter_lines():
            if chunk:
                data = json.loads(chunk.decode("utf-8")[5:])["token"]["text"]
                if data:
                    gen_text += data 
        return gen_text

if __name__ == '__main__':

    # client = OpenAI(
    #     api_key="EMPTY",
    #     base_url="http://10.210.9.11:32011/v1",
    # )
    # print("call api...")
    # messages=[
    #     {'role': 'user', 
    #     'content': [
    #         {'type': 'text', 'text': 'hello, who are you?'},
    #     ]}
    # ]
    # completion = client.chat.completions.create(
    #     model='Qwen3-VL-8B-Instruct',
    #     messages=messages,
    #     temperature=0,
    # )
    # raw = completion.to_json()
    # data = json.loads(raw)
    # content = data["choices"][0]["message"]["content"]
    # print(f'received output: {content}')

    # url = f'http://10.210.6.10:20903/generate_stream'
    # print(f'api test: url={url}')

    # time_interval = 30
    # max_retry = 20
    # successed = False
    # for i in range(max_retry):
    #     print(f'retry i={i}')
    #     try:
    #         out = api_request(
    #             url,
    #             stream=True,
    #             messages=[{'role': 'user', 'content': 'hello, who are you?'}],
    #             max_new_tokens=512
    #         )
    #         print(f'received output: {out}')
    #         successed = True
    #         break
    #     except:
    #         print(f'error: {traceback.format_exc()}')
    #         time.sleep(30)
    # if successed:
    #     exit(0)
    # else:
    #     exit(-1)

    client = OpenAI(
        api_key="sk-msxoHjQv4oPTtngKR9VzDLQn4WQ1Ge9b5H12YHUF7aeosivs",
        base_url="https://api.ppchat.vip/v1"
    )

    print("开始测试对话...")

    response = client.chat.completions.create(
        model="gemini-3-pro-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Hi, who are you?"
            }
        ]
    )
    print(response.choices[0].message.content)


    # client = OpenAI(
    #     api_key="495e7f4ae82ddc5ccdb928b1bb686375",
    #     base_url="https://dl.yunstorm.com/v1"
    # )

    # print("开始测试对话...")

    # response = client.chat.completions.create(
    #     model="qwen3-vl-235b-a22b-thinking",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {
    #             "role": "user",
    #             "content": "Hi, who are you?"
    #         }
    #     ],
    #     stream=True,
    # )

    # print(response)

    # final_text = ""
    # for chunk in response:
    #     if not getattr(chunk, "choices", None):
    #         continue
    #     if not chunk.choices:
    #         continue
    #     delta = getattr(chunk.choices[0], "delta", None)
    #     if not delta:
    #         continue
    #     for k in ("content", "reasoning", "reasoning_content", "text"):
    #         v = getattr(delta, k, None)
    #         if v:
    #             final_text += v
    #             break
    # print(final_text)
