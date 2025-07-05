import os
import numpy as np
import pandas as pd
import shutil
import argparse
import openai  # Make sure to install this package
from openai import OpenAI
import random
import time
import csv
import json
from tqdm import tqdm
import re
import base64
import requests
import io
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Initialize API
api_key = 'sk-X'
api_key_list = ['sk-X',
                'sk-X']

# TOTAL_DIR = '/mnt/workspace/zpf/OpenING'
TOTAL_DIR = "D:\\OneDrive\\mllm-eval\\OpenING"
OUTPUT_DIR = 'E:\\Users\Lance\\BaiduNetdiskDownload\\Gemini1.5+Flux_RAG-dev'  # Define your output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_and_load_json(content):
    
    input_text_list = []
    input_image_list = []
    onput_text_list = []
    output_image_list = []

    for input_step, input_content in enumerate(content['conversations'][0]['input']):
        input_text_list.append(input_content['text'].strip())
        input_image_list.append(input_content['image'])

    for output_step, output_content in enumerate(content['conversations'][1]['output']):
        onput_text_list.append(output_content['text'].strip())
        output_image_list.append(output_content['image'])

    return input_text_list, input_image_list, onput_text_list, output_image_list

def load_data(data_path):
    ori_data = []  # 初始化数据项列表
    io_data = []

    with open(data_path, encoding='utf-8') as file:  # 打开数据文件
        for line in tqdm(file):  # 遍历每一行
            content = json.loads(line)
            ori_data.append(content)  # 将每行数据加载为JSON对象并添加到列表
            # get input list etc. and return 5 lists
            ainput_list, ainput_image_list, aoutput_list, aoutput_image_list = parse_and_load_json(content)
            io_data.append({"input_text": ainput_list, "input_image": ainput_image_list, "output_text": aoutput_list, "output_image": aoutput_image_list})
    return ori_data, io_data

def save_results(real_data_item, generated_text_list, image_out_list):
    data_uid = real_data_item["total_uid"]
    # Save generated text as JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, f'{data_uid}.jsonl')

    saved_json = real_data_item.copy()
    if 'conversations' in saved_json and len(saved_json['conversations']) > 1:
        saved_json['conversations'][1]['output'] = []

    for index in range(len(generated_text_list)):
        a_out_item = {"text": generated_text_list[index].replace("**", "")}
        if index < len(image_out_list):
            if image_out_list[index] is not None:
                a_out_item["image"] = f'{data_uid}-o-{index}.jpg'
                # image_path = os.path.join(OUTPUT_DIR, f'{data_uid}-o-{index}.jpg')
                # image_out_list[index].save(image_path)
            else:
                a_out_item["image"] = None
        else:
            a_out_item["image"] = None

        saved_json['conversations'][1]['output'].append(a_out_item)

    with open(jsonl_path, mode='w', encoding='utf-8') as writer:
        json.dump(saved_json, writer, indent=4)


# Define a function to get paraphrases using OpenAI API
def get_gpt4answer(input_text, input_images, fewshot_examples):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    my_message = [{"role": "system", "content": SYSTEM_MESSAGE}]

    content = []
    for i in range(len(input_text)):
        content.append({"type": "text", "text": input_text[i]})
        if i < len(input_images) and input_images[i] != None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_images[i]}"}})

    for sample in fewshot_examples:
        my_message.append({"role":"user", "content": content})
        my_message.append({"role":"assistant", "content": sample})

    my_message.append({"role": "user", "content": content})
    payload = {"model": "gemini-1.5-pro-latest", "messages": my_message}
    # payload = {"model": "gpt-4o-mini", "messages": my_message}

    answer = ''
    while True:
        # try:
        response = requests.post("https://api.claudeshop.top/v1/chat/completions", headers=headers, json=payload)
        # Parse the response to extract the generated answer
        if response.status_code == 200:
            response_json = response.json()  # Parse the response as JSON
            print('reference:')
            print(fewshot_examples)
            print('real:')
            print(response_json)
            # Assuming the API response contains a 'choices' field with a list of completions
            answer = response_json['choices'][0]['message']['content']
            if not answer:
                continue
            if '<IMG>' not in answer:
                answer += ' <IMG>'
            return answer
        else:
            print(f"API request failed with status code {response.status_code}")
            print('reference:')
            print(fewshot_examples)
            print('real:')
            print(response)
            time.sleep(0.5)
            my_message = [{"role": "system", "content": SYSTEM_MESSAGE}]
            my_message.append({"role": "user", "content": content})
            payload = {"model": "gemini-1.5-pro-latest", "messages": my_message}
            continue
        # except Exception as e:
        #     print(f"Error parsing response: {e}")
        #     time.sleep(1)
        #     continue

    # result = openai.chat.completions.create(model="gpt-4o-2024-08-06",
    #                         messages=my_message)
    # answers = result.choices[0].message.content
    
    # result = openai.Completion.create(
    #     engine="gpt-4-1106-preview",  # You can use other engines as well
    #     prompt=f"Paraphrase the question '{question}' in a different way:",
    #     temperature=0.6,
    #     max_tokens=100,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    #     n=5  # Number of different answers you want to generate
    # )

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def decode_image(encoded_string):
    # 解码 base64 字符串
    image_data = base64.b64decode(encoded_string)
    # 将解码后的数据转换为图像格式
    image = Image.open(BytesIO(image_data))
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-path', type=str, default=TOTAL_DIR, help="Folder where the CSV and the images were downloaded.")
    # parser.add_argument('--data-file-name', type=str, default="model_test_data/Gemini1.5+Flux_test.jsonl", help="Folder where the CSV and the images were downloaded.")
    parser.add_argument('--data-file-name', type=str, default="dev_data.jsonl", help="Folder where the CSV and the images were downloaded.")
    args = parser.parse_args()
    data_path = os.path.join(args.meta_path, args.data_file_name)
    
    # edit_subtask_names = []
    # with open(os.path.join(args.meta_path, 'edit_image_input.txt')) as file:
    #     for i in file.readlines():
    #         edit_subtask_names.append(i.strip())

    saved_id = []
    for file in os.listdir(OUTPUT_DIR):
        if '.jsonl' in file and file.split('.')[0] not in saved_id:
            saved_id.append(file.split('.')[0])

    real_data_list, io_dir_list = load_data(data_path)

    with open(os.path.join(args.meta_path, '..\\codes\\zero-shot_generation_system.txt'), 'r') as f:
        file_contents = f.read()
    SYSTEM_MESSAGE = str(file_contents)

    runned_noinputtask_list = []
    runned_inputtask_list = []
    count = 0
    for a_index, a_data in tqdm(enumerate(io_dir_list)):
        
        input_image_path = a_data['input_image']
        input_text_list = a_data['input_text']

        gt_out_step = len(a_data['output_text'])
        generated_image_list = []

        if real_data_list[a_index]['total_uid'] in saved_id:
            continue

        fewshot_examples = []
        content = []
        few_shot_text = ''
        for i in range(len(a_data['output_text'])):
            content.append({"type": "text", "text": a_data['output_text'][i].replace('<image>','<IMG>')})
        fewshot_examples.append(content)

        # try:
        input_images = []
        for img_path in input_image_path:
            if img_path:
                input_images.append(encode_image(os.path.join(TOTAL_DIR, img_path)))
            else:
                input_images.append(None)
        
        # Prepare the utterance with input text and previous outputs
        input_text_list[0] = f"The number of generated text-image pairs can be {gt_out_step}: " + input_text_list[0]

        output = get_gpt4answer(input_text_list, input_images, fewshot_examples)
        output = output.replace('<br>','').strip()
        # print(output)

        time.sleep(0.5)

        output_steps = [part.strip() for part in output.split('<IMG>') if part.strip()]
        if len(output_steps) > gt_out_step:
            output_steps = [output.replace(' <IMG> ','').replace(' <IMG>','').replace('<IMG> ','').replace('<IMG>','')]

        # Save the results for the current data item
        save_results(real_data_list[a_index], output_steps, generated_image_list)
        # print(f"Processed and saved results for UID: {real_data_list[a_index]['total_uid']}")
