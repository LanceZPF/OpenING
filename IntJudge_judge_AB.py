import os
import numpy as np
import pandas as pd
import shutil
import argparse
import random
import time
import json
from tqdm import tqdm
import re
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Please define these two paths first
# The ckpt_path is the path of the downloaded IntJudge model, and the preprocessor_path is the path of the preprocessor of Qwen2-VL-7B-Instruct
ckpt_path = "~/judges/IntJudge"
# preprocessor_path = "~/.cache/Qwen2-VL-7B-Instruct"

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     ckpt_path, torch_dtype="auto", device_map="auto"
# )

model = Qwen2VLForConditionalGeneration.from_pretrained(
    ckpt_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
# default processer
processor = AutoProcessor.from_pretrained(ckpt_path)

ROOT_DIR = './'
# RESULT_DIR is the path of the directory where the results of all interleaved generation results are saved, for each model named "MODEL_NAME" the output dir should be "MODEL_NAME_output"
RESULT_DIR = os.path.join(ROOT_DIR, "Interleaved_Arena")
# OPENING_DIR is the path of the OpenING benchmark
OPENING_DIR = os.path.join(ROOT_DIR, "OpenING-Benchmark")
PK_FILE_NAME = os.path.join(RESULT_DIR, "new_data_instance_modelAB.json")
# OUTPUT_FILE is the path of the file where the judge results are saved
OUTPUT_FILE = os.path.join(ROOT_DIR, "Interleaved_Arena", "intjudge_rag_modelAB_results.json") 

def generate_total_uid(meta_task_id, subtask_id, data_id):
    # Ensure meta_task_id is 2 digits, subtask_id is 2 digits, and data_id is 3 digits
    return f'{int(meta_task_id):02}{int(subtask_id):02}{int(data_id):03}'

pattern = r"/(\d+)-([io]-\d+\.jpg)$"
file_path = os.path.join(OPENING_DIR, "test_data.jsonl")
# Read the JSONL file line by line
image_to_uid = {}
with open(file_path, "r") as file:
    for line in file:
        # Parse each line as a JSON object
        data = json.loads(line.strip())
        
        # Extract total_uid and conversation details

        conversations = data.get("conversations", [])
        
        # Iterate over the conversations to find images
        for conversation in conversations:
            for output in conversation.get("input", []):
                image_path = output.get("image")
                # Check if image path is not None and add to dictionary
                meta_task_id = data['meta_task_id']
                subtask_id = data['subtask_id']
                data_id = data['data_id']
                if image_path:
                    match = re.search(pattern, image_path)
                    if match:
                        # Extract the data_id from the first capturing group
                        data_id = match.group(1)
                    total_uid = generate_total_uid(meta_task_id, subtask_id, data_id)
                    image_to_uid[image_path] = total_uid

def generate_total_uid(meta_task_id, subtask_id, data_id):
    # Ensure meta_task_id is 2 digits, subtask_id is 2 digits, and data_id is 3 digits
    return f'{int(meta_task_id):02}{int(subtask_id):02}{int(data_id):03}'

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

    my_message = [{"role": "system", "content": SYSTEM_MESSAGE}]

    content = []
    input_text[0] = "<INPUT>: " + input_text[0]
    for i in range(len(input_text)):
        content.append({"type": "text", "text": input_text[i].replace("<BEGIN>","")})
        if i < len(input_images) and input_images[i] != None:
            content.append({"type": "image", "image": input_images[i]})

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
            content.append({"type": "image", "image": modelA_output_images[i]})

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
            content.append({"type": "image", "image": modelB_output_images[i]})

    content.append({"type": "text", "text": "\nPlease directly output \"A is better\" or \"B is better\" or \"Tie(A is better)\" or \"Tie(B is better)\": \n"})
    my_message.append({"role": "user", "content": content})

    text = processor.apply_chat_template(
        my_message, tokenize=False, add_generation_prompt=True
    )
    try:
        image_inputs, video_inputs = process_vision_info(my_message)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
    except Exception as e:
        print(e)
        print(content)
        return False
    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

def save_judge_results(judge_results):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(judge_results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # with open(os.path.join(OPENING_DIR, 'qwen_judge_system.txt'), 'r') as f:
    #     file_contents = f.read()
    # SYSTEM_MESSAGE = str(file_contents)
    SYSTEM_MESSAGE = "You are an impartial judge in multimodal content evaluation. Given a referenced input question (marked by <INPUT>: ), your task is to compare the quality of the answers generated by model A (marked by <OUTPUT_A>: ) and model B (marked by <OUTPUT_B>: ), and judge which one is better.\n"
    
    total_pk_list = load_pk_file()
    judge_results = load_judge_results()
    runned_id = []
    for i in judge_results:
        runned_id.append((i['data_id'],i['model_A']['id'],i['model_B']['id']))

    false_count = 0

    for index, pk_data in tqdm(enumerate(total_pk_list),total=len(total_pk_list)):
        current_data_uid = pk_data['data_id']
        current_total_index = (current_data_uid,pk_data['model_A']['id'],pk_data['model_B']['id'])
        if current_total_index in runned_id:
            continue
        
        current_model_A = pk_data['model_A']['name']
        current_model_B = pk_data['model_B']['name']
        current_file_path1 = os.path.join(RESULT_DIR, f"{current_model_A}_output", f'{current_data_uid}.json')
        current_file_path2 = os.path.join(RESULT_DIR, f"{current_model_B}_output", f'{current_data_uid}.json')

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
        assert a_data1['input_text'] == a_data2['input_text']

        input_text_list = a_data1['input_text']

        img_count = -1

        modelA_output_images = []
        modelB_output_images = []
        input_images = []

        for i in range(len(input_text_list)):
            if a_data1['input_image'][i] != None:
                img_count += 1
                input_text_list[i] = input_text_list[i].replace('<image>', '') + f" <IMG_{img_count}>" + f"</IMG_{img_count}>"
            else:
                input_text_list[i] = input_text_list[i].replace('<image>', '') 

        # For each input text, get the GPT-4 generated answer
        for img_path in a_data1['input_image']:
            if img_path:
                # if './images/' in img_path:
                #     image = img_path
                #     img_tem = img_path
                #     temp_uid = image_to_uid[img_tem]
                #     new_image_path = re.sub(pattern, rf"/{temp_uid}-\2", image)
                #     new_img_name = new_image_path.split('/')[-1]
                # else:
                #     new_img_name = img_path
                # real_new_path = os.path.join(INPUT_DIR, new_img_name)
                real_new_path = os.path.join(OPENING_DIR, img_path)
                input_images.append(real_new_path)
            else:
                input_images.append(None)

        for i in range(len(a_data1['output_text'])):
            if a_data1['output_image'][i] != None:
                temp_img_path = os.path.join(RESULT_DIR, f"{current_model_A}_output", a_data1['output_image'][i].split('/')[-1])
                if os.path.exists(temp_img_path):
                    img_count += 1
                    a_data1['output_text'][i] = a_data1['output_text'][i].replace('<image>','') + f" <IMG_{img_count}>" + f"</IMG_{img_count}>"
                    modelA_output_images.append(temp_img_path)
                else:
                    a_data1['output_text'][i] = a_data1['output_text'][i].replace('<image>','')
                    modelA_output_images.append(None)
            else:
                a_data1['output_text'][i] = a_data1['output_text'][i].replace('<image>','')
                modelA_output_images.append(None)

        for i in range(len(a_data2['output_text'])):
            if a_data2['output_image'][i] != None:
                temp_img_path = os.path.join(RESULT_DIR, f"{current_model_B}_output", a_data2['output_image'][i].split('/')[-1])
                if os.path.exists(temp_img_path):
                    img_count += 1
                    a_data2['output_text'][i] = a_data2['output_text'][i].replace('<image>','') + f" <IMG_{img_count}>" + f"</IMG_{img_count}>"
                    modelB_output_images.append(temp_img_path)
                else:
                    a_data2['output_text'][i] = a_data2['output_text'][i].replace('<image>','')
                    modelB_output_images.append(None)
            else:
                a_data2['output_text'][i] = a_data2['output_text'][i].replace('<image>','')
                modelB_output_images.append(None)

        output = get_gpt4answer(input_text_list, input_images, a_data1['output_text'], modelA_output_images, a_data2['output_text'], modelB_output_images)
        if output == False:
            false_count += 1
            print(false_count)
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
