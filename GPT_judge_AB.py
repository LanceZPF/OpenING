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
import random
import random

# Initialize API
api_key = "*****"

ROOT_DIR = './'
OPENING_DIR = os.path.join(ROOT_DIR, "OpenING-Benchmark")
# INPUT_DIR = os.path.join(ROOT_DIR, "InputImages")

PK_FILE_NAME = './Interleaved_Arena/new_data_instance_modelAB.json'
OUTPUT_FILE = "./Interleaved_Arena/gpt-judge_results.json"  # Define your output directory

def load_judge_results():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            judge_results = json.load(f)
    else:
        judge_results = []
    return judge_results

def load_pk_file():
    with open(os.path.join(PK_FILE_NAME), 'r') as f:
        pk_data = json.load(f)
        pk_list = []
        for data in pk_data:
            # if data['model_A']['name'] in ['SEED-X', 'Show-o'] and data['model_B']['name'] in ['SEED-X', 'Show-o']:
            #     pk_list.append(data)
            # if data['model_A']['name'] != "SEED-LLaMA" and data['model_B']['name'] != "SEED-LLaMA":
            pk_list.append(data)
        return pk_list

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
    with open(data_path, encoding='utf-8') as file:  # 打开数据文件
        content = json.load(file)
        ori_data = (content)  # 将每行数据加载为JSON对象并添加到列表
        # get input list etc. and return 5 lists
        ainput_list, ainput_image_list, aoutput_list, aoutput_image_list = parse_and_load_json(content)
        io_data = {"input_text": ainput_list, "input_image": ainput_image_list, "output_text": aoutput_list, "output_image": aoutput_image_list}
    return ori_data, io_data

# Define a function to get paraphrases using OpenAI API
def get_gpt4answer(input_text, input_images, modelA_output_textl, modelA_output_images, modelB_output_textl, modelB_output_images):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }

    my_message = [{"role": "system", "content": SYSTEM_MESSAGE}]

    content = []
    input_text[0] = "<INPUT>: " + input_text[0]
    for i in range(len(input_text)):
        content.append({"type": "text", "text": input_text[i].replace("<BEGIN>","")})
        if i < len(input_images) and input_images[i] != None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_images[i]}"}})

    if len(modelA_output_textl) >= 1 and modelA_output_textl[0]:
        modelA_output_textl[0] = "<OUTPUT_A>: " + modelA_output_textl[0]
    elif len(modelA_output_textl) >= 1:
        modelA_output_textl[0] = "\n<OUTPUT_A>:"
    else:
        modelA_output_textl.append("\n<OUTPUT_A>:")

    for i in range(max(len(modelA_output_textl), len(modelA_output_images))):
        if i < len(modelA_output_textl):
            content.append({"type": "text", "text": modelA_output_textl[i]})
        if i < len(modelA_output_images) and modelA_output_images[i] != None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{modelA_output_images[i]}"}})
    if len(modelB_output_textl) >= 1 and modelB_output_textl[0]:
        modelB_output_textl[0] = "\n<OUTPUT_B>: " + modelB_output_textl[0]
    elif len(modelB_output_textl) >= 1:
        modelB_output_textl[0] = "\n<OUTPUT_B>:"
    else:
        modelB_output_textl.append("\n<OUTPUT_B>:")
    for i in range(max(len(modelB_output_textl), len(modelB_output_images))):
        if i < len(modelB_output_textl):
            content.append({"type": "text", "text": modelB_output_textl[i]})
        if i < len(modelB_output_images) and modelB_output_images[i] != None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{modelB_output_images[i]}"}})

    content.append({"type": "text", "text": "\nPlease only output which model is better: "})

    my_message.append({"role": "user", "content": content})

    payload = json.dumps({"model": "gpt-4o", "messages": my_message, "max_tokens": 5})

    answer = ''
    max_try = 0
    while True:
        response = requests.request("POST", url, headers=headers, data=payload)
        # Parse the response to extract the generated answer
        if response.status_code == 200:
            response_json = response.json()  # Parse the response as JSON
            print(response_json)
            # Assuming the API response contains a 'choices' field with a list of completions
            answer = response_json['choices'][0]['message']['content']
            if not answer:
                continue
            return answer
        else:
            print(f"API request failed with status code {response.status_code}")
            # print(payload)
            time.sleep(1)
            max_try += 1
            if max_try > 10:
                return False
            continue
            # return False

# Function to encode the image
def encode_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Resize the image to 512x512, maintaining aspect ratio if necessary
        img = img.resize((512, 512))
        # Save the image to a bytes buffer in JPEG format
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        
        # Encode the image to base64
        return base64.b64encode(img_bytes).decode('utf-8')
    
def decode_image(encoded_string):
    # 解码 base64 字符串
    image_data = base64.b64decode(encoded_string)
    # 将解码后的数据转换为图像格式
    image = Image.open(BytesIO(image_data))
    image.save('temp.jpg')
    return image

def save_judge_results(judge_results):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(judge_results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    with open(os.path.join(OPENING_DIR, 'judge_system.txt'), 'r') as f:
        file_contents = f.read()
    SYSTEM_MESSAGE = str(file_contents)

    total_pk_list = load_pk_file()
    judge_results = load_judge_results()
    runned_id = []
    for i in judge_results:
        runned_id.append((i['data_id'],i['model_A']['id'],i['model_B']['id']))

    for index, pk_data in enumerate(total_pk_list):
        current_data_uid = pk_data['data_id']
        current_total_index = (current_data_uid,pk_data['model_A']['id'],pk_data['model_B']['id'])

        if current_total_index in runned_id:
            continue
        
        current_model_A = pk_data['model_A']['name']
        current_model_B = pk_data['model_B']['name']
        current_file_path1 = os.path.join(ROOT_DIR, f"{current_model_A}_output", f'{current_data_uid}.json')
        current_file_path2 = os.path.join(ROOT_DIR, f"{current_model_B}_output", f'{current_data_uid}.json')

        try:
            ori_data1, a_data1 = load_data(current_file_path1)
            ori_data2, a_data2 = load_data(current_file_path2)
        except Exception as e:
            print(f"ERROR: {e}")
            print(current_file_path1, current_file_path2)
            continue

        # if len(a_data1['output_text']) < 1 or len(a_data2['output_text']) < 1:
        #     continue

        if a_data1['input_image'] != a_data2['input_image']:
            if len(a_data1['input_image']) > len(a_data2['input_image']):
                input_image_list = a_data1['input_image']
            else:
                input_image_list = a_data2['input_image']
        else:
            input_image_list = a_data1['input_image']
        # assert a_data1['input_text'] == a_data2['input_text']

        input_text_list = a_data1['input_text']

        img_count = -1

        modelA_output_images = []
        modelB_output_images = []
        input_images = []

        for i in range(len(input_text_list)):
        # For each input text, get the GPT-4 generated answer
            img_path = a_data1['input_image'][i]
            if img_path:
                temp_img_path = os.path.join(OPENING_DIR, img_path)
                # if len(img_path) > 16:
                #     temp_img_path = os.path.join(OPENING_DIR, img_path)
                # else:
                #     temp_img_path = os.path.join(INPUT_DIR, img_path)
                try:
                    input_images.append(encode_image(temp_img_path))
                    img_count += 1
                    input_text_list[i] = input_text_list[i].replace('<BEGIN>','').replace('<image>', '') + f" <IMG_{img_count}>" + f"</IMG_{img_count}>"
                except Exception as e:
                    print(f"ERROR: {e}")
                    print(temp_img_path)
                    input_images.append(None)
                    input_text_list[i] = input_text_list[i].replace('<BEGIN>','').replace('<image>', '') 
            else:
                input_images.append(None)
                input_text_list[i] = input_text_list[i].replace('<BEGIN>','').replace('<image>', '') 

        for i in range(len(a_data1['output_text'])):
            if a_data1['output_image'][i] != None:
                temp_img_path = os.path.join(ROOT_DIR, f"{current_model_A}_output", a_data1['output_image'][i].split('/')[-1])
                if os.path.exists(temp_img_path):
                    try:
                        modelA_output_images.append(encode_image(temp_img_path))
                        img_count += 1
                        a_data1['output_text'][i] = a_data1['output_text'][i].replace('<image>','') + f" <IMG_{img_count}>" + f"</IMG_{img_count}>"
                    except Exception as e:
                        print(f"ERROR: {e}")
                        print(temp_img_path)
                        a_data1['output_text'][i] = a_data1['output_text'][i].replace('<image>', '')
                        modelA_output_images.append(None)
                else:
                    a_data1['output_text'][i] = a_data1['output_text'][i].replace('<image>','')
                    modelA_output_images.append(None)
            else:
                a_data1['output_text'][i] = a_data1['output_text'][i].replace('<image>','')
                modelA_output_images.append(None)

        for i in range(len(a_data2['output_text'])):
            if a_data2['output_image'][i] != None:
                temp_img_path = os.path.join(ROOT_DIR, f"{current_model_B}_output", a_data2['output_image'][i].split('/')[-1])
                if os.path.exists(temp_img_path):
                    try:
                        modelB_output_images.append(encode_image(temp_img_path))
                        img_count += 1
                        a_data2['output_text'][i] = a_data2['output_text'][i].replace('<image>','') + f" <IMG_{img_count}>" + f"</IMG_{img_count}>"
                    except Exception as e:
                        print(f"ERROR: {e}")
                        print(temp_img_path)
                        a_data2['output_text'][i] = a_data2['output_text'][i].replace('<image>', '')
                        modelB_output_images.append(None)
                else:
                    a_data2['output_text'][i] = a_data2['output_text'][i].replace('<image>','')
                    modelB_output_images.append(None)
            else:
                a_data2['output_text'][i] = a_data2['output_text'][i].replace('<image>','')
                modelB_output_images.append(None)
        
        output = get_gpt4answer(input_text_list, input_images, a_data1['output_text'], modelA_output_images, a_data2['output_text'], modelB_output_images)
        if not output:
            print('FAIL')
            print(index)
            print(pk_data)
            print(input_text_list)
            print(input_images)
            print(a_data1['output_text'])
            print(len(modelA_output_images))
            print(a_data2['output_text'])
            print(len(modelB_output_images))
            continue
        current_data = pk_data.copy()
        if output[0] == "A" or output[0] == "B":
            current_data['winner'] = output[0]
        elif output[:3].lower() == "tie":
            current_data['winner'] = output[:5]+')'
        else:
            # random choise 'Tie(A)' or 'Tie(B)'
            current_data['winner'] = 'Tie('

            flag = False
            for t_i in output:
                if t_i == 'A':
                    current_data['winner'] += 'A)'
                    flag = True
                    break
                elif t_i == 'B':
                    current_data['winner'] += 'B)'
                    flag = True
                    break

            if not flag:
                if random.randint(0, 1) == 0:
                    current_data['winner'] += 'A)'
                else:
                    current_data['winner'] += 'B)'

        judge_results.append(current_data)
        save_judge_results(judge_results)
