import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
import argparse
import random
import time
import csv
import json
from tqdm import tqdm
import re
import base64

import torch

from diffusers import FluxPipeline, FluxImg2ImgPipeline
from diffusers.utils import load_image

device = "cuda"
seed = 42
model_id = "/mnt/workspace/zpf/.cache/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-schnell`

t2i_pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
i2i_pipe = FluxImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
t2i_pipe = t2i_pipe.to(device)
i2i_pipe = i2i_pipe.to(device)

TOTAL_DIR = '/mnt/workspace/zpf/OpenING'
# TOTAL_DIR = "D:\\OneDrive\\mllm-eval\\OpenING"
OUTPUT_DIR = './Gemini1.5+Flux_RAG-dev'  # Define your output directory
# OUTPUT_DIR = 'E:\\Users\Lance\\BaiduNetdiskDownload\\Gemini1.5+Flux_output'  # Define your output directory
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
                image_path = os.path.join(OUTPUT_DIR, f'{data_uid}-o-{index}.jpg')
                image_out_list[index].save(image_path)
            else:
                a_out_item["image"] = None
        else:
            a_out_item["image"] = None

        saved_json['conversations'][1]['output'].append(a_out_item)

    with open(jsonl_path, mode='w', encoding='utf-8') as writer:
        json.dump(saved_json, writer, indent=4)


# Define a function to get paraphrases using OpenAI API
def get_saved_answer(total_uid):
    with open(os.path.join(OUTPUT_DIR,total_uid+'.jsonl'), encoding='utf-8') as file:  # 打开数据文件
        content = json.load(file)
        ainput_list, ainput_image_list, aoutput_list, aoutput_image_list = parse_and_load_json(content)
    return aoutput_list

# Function to encode the image
def encode_image(image_path):
    image = load_image(image_path).resize((1024, 1024))
    return image

# Define a function to generate image using DALL-E API
def generate_image(a_real_data, input_text, output_text, i_step, input_images=[]):
    prompt = f"The prompt for this generation is: {output_text}. The context of this task is: {input_text}."

    if len(input_images) > 0:
        init_image = input_images[0]
        # image = i2i_pipe(
        #     prompt=prompt,
        #     image=init_image,
        #     height=1024,
        #     width=1024,
        #     strength=0.95,
        #     guidance_scale=0.0,
        #     num_inference_steps=50,
        #     max_sequence_length=512,
        #     generator=torch.Generator("cpu").manual_seed(0)
        # ).images[0]
        image = i2i_pipe(prompt=prompt, image=init_image, num_inference_steps=4, strength=0.95, guidance_scale=0.0, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    else:
        # use a larger number if you are using [dev] 
        image = t2i_pipe(prompt, output_type="pil", num_inference_steps=4, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        # image = t2i_pipe(
        #     prompt,
        #     height=1024,
        #     width=1024,
        #     guidance_scale=3.5,
        #     num_inference_steps=50,
        #     max_sequence_length=512,
        #     generator=torch.Generator("cpu").manual_seed(0)
        # ).images[0]
    
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
        if '.jpg' in file and file.split('.')[0] not in saved_id:
            saved_id.append(file.split('.')[0])

    real_data_list, io_dir_list = load_data(data_path)

    # with open(os.path.join(args.meta_path, '..\\codes\\zero-shot_generation_system.txt'), 'r') as f:
    #     file_contents = f.read()
    # SYSTEM_MESSAGE = str(file_contents)

    runned_noinputtask_list = []
    runned_inputtask_list = []
    count = 0
    for a_index, a_data in tqdm(enumerate(io_dir_list), total=len(io_dir_list)):
        
        input_image_path = a_data['input_image']
        input_text_list = a_data['input_text']

        gt_out_step = len(a_data['output_text'])
        generated_image_list = []
        # For each input text, get the GPT-4 generated answer

        if real_data_list[a_index]['total_uid'] in saved_id:
            continue

        input_images = []
        for img_path in input_image_path:
            if img_path:
                input_images.append(encode_image(os.path.join(TOTAL_DIR, img_path)))
                break
        
        # Prepare the utterance with input text and previous outputs

        # input_image_list = []
        # for img in input_image_path:
        #     if img != None:
        #         input_image_list.append(img)

        output_steps = get_saved_answer(real_data_list[a_index]['total_uid'])

        gen_img_num = min(gt_out_step,len(output_steps))
        for i_step in range(gen_img_num):
            saved_path = generate_image(real_data_list[a_index], input_text_list[0], output_steps[i_step], i_step, input_images=input_images)
            if saved_path:
                generated_image_list.append(saved_path)

        # Save the results for the current data item
        save_results(real_data_list[a_index], output_steps, generated_image_list)
        torch.cuda.empty_cache()
        print(f"Processed and saved results for UID: {real_data_list[a_index]['total_uid']}")
