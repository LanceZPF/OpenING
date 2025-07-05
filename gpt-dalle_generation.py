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

# Initialize OpenAI API
api_key = "*****"

TOTAL_DIR = "./OpenING-Benchmark"
OUTPUT_DIR = './gen_outputs/GPT-4o+DALL-E3_output'
# OUTPUT_DIR = 'GPT-4o+DALL-E3_dumb-dev'  # Define your output directory
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
    jsonl_path = os.path.join(OUTPUT_DIR, f'{data_uid}.json')

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


# Define a function to get paraphrases using OpenAI API or using API from third-party service providers
def get_gpt4answer(input_text, input_images):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    my_message = [{"role": "system", "content": SYSTEM_MESSAGE}]

    # for sample in FEW_SHOT_EXAMPLES:
    #     my_message.append({"role":"user", "content": sample['context']})
    #     my_message.append({"role":"assistant", "content": sample['response']} )

    content = []
    for i in range(len(input_text)):
        content.append({"type": "text", "text": input_text[i].replace('semi-naked','no ')})
        if i < len(input_images) and input_images[i] != None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_images[i]}"}})

    my_message.append({"role": "user", "content": content})
    payload = {"model": "gpt-4o", "messages": my_message}

    answer = ''
    while True:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # Parse the response to extract the generated answer
        if response.status_code == 200:
                response_json = response.json()  # Parse the response as JSON
                # Assuming the API response contains a 'choices' field with a list of completions
                answer = response_json['choices'][0]['message']['content']
                print(input_text)
                print(answer)
                if not answer:
                    continue
                if '<IMG>' not in answer:
                    answer += '<IMG>'
                if 'I\'m sorry' in answer:
                    continue
                return answer
        else:
            print(f"API request failed with status code {response.status_code}")
            time.sleep(1)
            continue

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

# 使用 matplotlib 显示图像
def show_image(image):
    plt.imshow(image)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()

# Define a function to generate image using DALL-E API
def generate_image(a_real_data, input_text, output_text, i_step):
    # Define the API request URL and headers
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    four_digit_num = str(i_step).zfill(4)

    prompt = f"Please generate images using seed 5000. The context of this task is: {input_text[0]}. The prompt for this generation is: {output_text}. Seed:5000. - {four_digit_num}"
    max_try = 0
    while True:
        # Define the payload
        payload = {
            "model": "dall-e-3",
            "prompt": prompt,
            "size": "1024x1024",
            "quality": "standard",
            "n": 1  # Generate one image
        }

        try:
            # Make the request to the OpenAI API
            response = requests.post(url, headers=headers, json=payload)
            # Check if the request was successful
            if response.status_code == 200:

                response_json = response.json()
                print(response_json)
                image_url = response_json['data'][0]['url']  # Extract the image URL

                # Download and save the image
                image_response = requests.get(image_url, stream=True)
                if image_response.status_code == 200:
                    save_path_name = os.path.join(OUTPUT_DIR,f'{a_real_data["total_uid"]}-o-{i_step}.jpg')
                    # save_path_name = os.path.join(f'{a_real_data["total_uid"]}-o-{i_step}.jpg')
                    with open(save_path_name, 'wb') as f:
                        image_response.raw.decode_content = True
                        shutil.copyfileobj(image_response.raw, f)
                    return f'{a_real_data["total_uid"]}-o-{i_step}.jpg'
                else:
                    print(f"Failed to download image, status code: {image_response.status_code}")
                    time.sleep(1)
                    continue
            else:
                print(f"DALLE API request failed with status code {response.status_code}")
                # prompt = re.sub(r'[^a-zA-Z0-9\s]', '', output_text)
                output_text = output_text[:len(output_text)//2]
                prompt = f"Very safe information, just generate an illustration: {output_text}"
                print(prompt)
                time.sleep(1)
                max_try += 1
                if max_try > 10:
                    return False
                continue
        except Exception as e:
            print(f"Error parsing DALLE response: {e}")
            time.sleep(1)
            max_try += 1
            if max_try > 10:
                return False
            continue

# Define a function to generate image using DALL-E API with image input
def generate_image_with_input(a_real_data, input_text, output_text, i_step, input_image_path_list=None):
    # Define the API request URL and headers
    url = "https://api.openai.com/v1/images/edits"

    # Initialize the prompt and prepare image input
    prompt = f"{input_text[0]}"

    # If an image is provided, encode it and add it to the prompt
    if input_image_path_list:
        image_path = os.path.join(TOTAL_DIR, input_image_path_list[0])

    img = Image.open(image_path)
    
    # Convert to PNG
    png_path = image_path.split('/')[-1].split('.jpg')[0] + ".png"

    img = img.convert("RGBA")
    img.save(png_path, format="PNG")
    
    # Check if the image is greater than 4MB
    if os.path.getsize(png_path) > 4 * 1024 * 1024:
        # Compress the PNG by adjusting the compression level
        img = Image.open(png_path)
        img = img.convert("RGBA")
        img.save(png_path, format="PNG", optimize=True, compress_level=9)

    # Prepare the files and data for the POST request
    with open(png_path, "rb") as image_file:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        files = {
            "image": image_file
        }
        data = {
            "prompt": prompt,
            "n": 2,
            "size": "1024x1024"
        }

        # Send the request
        response = requests.post(url, headers=headers, files=files, data=data)
        print(response.json())

        # Handle the response
        if response.status_code == 200:
            result = response.json()
            image_url = result['data'][0]['url']
            img_data = requests.get(image_url).content
            with open(png_path.replace('-i-','-o-'), 'wb') as handler:
                handler.write(img_data)
        else:
            print(f"Error: {response.status_code}, {response.text}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-path', type=str, default=TOTAL_DIR, help="Folder where the CSV and the images were downloaded.")
    # parser.add_argument('--data-file-name', type=str, default="model_test_data/GPT-4o+DALL-E3_test.jsonl", help="Folder where the CSV and the images were downloaded.")
    parser.add_argument('--data-file-name', type=str, default="test_data.jsonl", help="Folder where the CSV and the images were downloaded.")
    args = parser.parse_args()
    data_path = os.path.join(args.meta_path, args.data_file_name)

    saved_id = []
    for file in os.listdir(OUTPUT_DIR):
        if '.jpg' in file and file.split('-')[0] not in saved_id:
            saved_id.append(file.split('-')[0])

    real_data_list, io_dir_list = load_data(data_path)

    with open('./prompts/zero-shot_generation_system.txt', 'r') as f:
        file_contents = f.read()
    SYSTEM_MESSAGE = str(file_contents)

    count = 0
    for a_index, a_data in tqdm(enumerate(io_dir_list)):
        
        input_image_path = a_data['input_image']
        input_text_list = a_data['input_text']

        gt_out_step = len(a_data['output_text'])
        generated_image_list = []
        # For each input text, get the GPT-4 generated answer

        if real_data_list[a_index]['total_uid'] in saved_id:
            continue

        # try:
        input_images = []
        for img_path in input_image_path:
            if img_path:
                input_images.append(encode_image(os.path.join(TOTAL_DIR, img_path)))
            else:
                input_images.append(None)
        
        # Prepare the utterance with input text and previous outputs
        input_text_list[0] = f"The number of generated text-image pairs can be {gt_out_step}: " + input_text_list[0]

        input_image_list = []
        for img in input_image_path:
            if img != None:
                input_image_list.append(img)

        output = get_gpt4answer(input_text_list, input_images)

        # time.sleep(1)

        output_steps = [part for part in output.split('<IMG>') if part]

        gen_img_num = min(gt_out_step,len(output_steps))
        print(gen_img_num)
        for i_step in range(gen_img_num):
            saved_path = generate_image(real_data_list[a_index], input_text_list, output_steps[i_step], i_step)
            # if len(input_image_list) > 0:
            #     saved_path = generate_image_with_input(real_data_list[a_index], input_text_list, output_steps[i_step], i_step, input_image_list)
            # else:
            #     saved_path = generate_image(real_data_list[a_index], input_text_list, output_steps[i_step], i_step)
            if saved_path != False:
                generated_image_list.append(saved_path)

        # Save the results for the current data item
        save_results(real_data_list[a_index], output_steps, generated_image_list)
        print(f"Processed and saved results for UID: {real_data_list[a_index]['total_uid']}")
