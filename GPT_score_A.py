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
import glob

# Initialize API
api_key = "sk-XXX"

ROOT_DIR = './'
OPENING_DIR = os.path.join(ROOT_DIR, "OpenING-Benchmark")
# INPUT_DIR = os.path.join(ROOT_DIR, "InputImages")
PK_FILE_NAME = "./Interleaved_Arena/data_instance_modelAB_new.json"
OUTPUT_FILE = "./Interleaved_Arena/gpt-score_results_new.json"  # Define your output directory

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

def get_gpt4answer(input_text, input_images, modelA_output_textl, modelA_output_images):
    url = "http://35.220.164.252:3888/v1/chat/completions"
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }

    my_message = [{"role": "system", "content": SYSTEM_MESSAGE}]

    content = []
    input_text[0] = "INPUT: " + input_text[0]
    for i in range(len(input_text)):
        content.append({"type": "text", "text": input_text[i].replace("<BEGIN>","")})
        if i < len(input_images) and input_images[i] != None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_images[i]}"}})

    if len(modelA_output_textl) >= 1 and modelA_output_textl[0]:
        modelA_output_textl[0] = "\nOUTPUT: " + modelA_output_textl[0]
    elif len(modelA_output_textl) >= 1:
        modelA_output_textl[0] = "\nOUTPUT: "
    else:
        modelA_output_textl.append("\nOUTPUT: None")

    for i in range(max(len(modelA_output_textl), len(modelA_output_images))):
        if i < len(modelA_output_textl):
            content.append({"type": "text", "text": modelA_output_textl[i]})
        if i < len(modelA_output_images) and modelA_output_images[i] != None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{modelA_output_images[i]}"}})

    content.append({"type": "text", "text": "\nPlease only output the json result: "})

    my_message.append({"role": "user", "content": content})

    functions = [
        {
            "name": "evaluate_multimodal_content",
            "description": "Evaluate the quality of interleaved image-text content based on specific criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "Correctness": {
                        "type": "object",
                        "properties": {
                            "Score": {"type": "integer"},
                            "Justification": {"type": "string"}
                        },
                        "required": ["Score", "Justification"]
                    },
                    "Image-Text Coherency": {
                        "type": "object",
                        "properties": {
                            "Score": {"type": "integer"},
                            "Justification": {"type": "string"}
                        },
                        "required": ["Score", "Justification"]
                    },
                    "Multi-step Consistency": {
                        "type": "object",
                        "properties": {
                            "Score": {"type": "integer"},
                            "Justification": {"type": "string"}
                        },
                        "required": ["Score", "Justification"]
                    },
                    "Content Quality": {
                        "type": "object",
                        "properties": {
                            "Score": {"type": "integer"},
                            "Justification": {"type": "string"}
                        },
                        "required": ["Score", "Justification"]
                    },
                    "Human Preference Alignment": {
                        "type": "object",
                        "properties": {
                            "Score": {"type": "integer"},
                            "Justification": {"type": "string"}
                        },
                        "required": ["Score", "Justification"]
                    },
                    "Completeness": {
                        "type": "object",
                        "properties": {
                            "Score": {"type": "integer"},
                            "Justification": {"type": "string"}
                        },
                        "required": ["Score", "Justification"]
                    },
                    "Content Richness": {
                        "type": "object",
                        "properties": {
                            "Score": {"type": "integer"},
                            "Justification": {"type": "string"}
                        },
                        "required": ["Score", "Justification"]
                    }
                },
                "required": [
                    "Correctness", "Image-Text Coherency", "Multi-step Consistency", 
                    "Content Quality", "Human Preference Alignment", 
                    "Completeness", "Content Richness"
                ]
            }
        }
    ]

    payload = json.dumps({
        "model": "gpt-4o",
        "messages": my_message,
        "functions": functions,
        "function_call": {"name": "evaluate_multimodal_content"}
    })
    # payload = json.dumps({"model": "gpt-4o", "messages": my_message, "max_tokens": 800})

    final_answer = ''
    max_try = 0
    while True:
        response = requests.request("POST", url, headers=headers, data=payload)
        # Parse the response to extract the generated answer
        if response.status_code == 200:
            response_json = response.json()  # Parse the response as JSON
            print(response_json['choices'][0]['message'])
            # Assuming the API response contains a 'choices' field with a list of completions
            if 'function_call' in response_json['choices'][0]['message']:
                answer = response_json['choices'][0]['message']['function_call']['arguments']
            else:
                # Fall back to normal content if no function call is made
                answer = response_json['choices'][0]['message']['content']
            try:
                final_answer = json.loads(answer)
            except Exception as e:
                print(e)
                print(final_answer)
            if not final_answer or final_answer == '':
                continue
            return final_answer
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

def main_gpt_judge_score_by_pairwise_file_format():

    total_pk_list = load_pk_file()
    judge_results = load_judge_results()
    runned_id = []
    for i in judge_results:
        runned_id.append((i['data_id'],i['model']['id']))

    for index, pk_data in enumerate(total_pk_list):
        current_data_uid = pk_data['data_id']

        if (current_data_uid,pk_data['model_A']['id']) in runned_id and (current_data_uid,pk_data['model_B']['id']) in runned_id:
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
        assert a_data1['input_text'] == a_data2['input_text']

        input_text_list = a_data1['input_text']

        img_count = -1
        input_img_count = -1

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
                    input_img_count += 1
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
                        input_img_count += 1
                        a_data2['output_text'][i] = a_data2['output_text'][i].replace('<image>','') + f" <IMG_{input_img_count}>" + f"</IMG_{input_img_count}>"
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

        output = get_gpt4answer(input_text_list, input_images, a_data1['output_text'], modelA_output_images)
        print(f"A: {output}")
        current_data = pk_data.copy()
        # delete model_B key and its value
        current_data.pop('model_B', None)
        # rename the key model_A as model
        current_data['model'] = current_data.pop('model_A')
        current_data['score'] = output
        judge_results.append(current_data)

        output = get_gpt4answer(input_text_list, input_images, a_data2['output_text'], modelB_output_images)
        print(f"B: {output}")
        current_data = pk_data.copy()
        # delete model_A key and its value
        current_data.pop('model_A', None)
        # rename the key model_B as model
        current_data['model'] = current_data.pop('model_B')
        current_data['score'] = output
        judge_results.append(current_data)

        save_judge_results(judge_results)

def main_evaluate_output_directory(output_dir):
    """
    Evaluate all model outputs in the specified directory using GPT scoring.
    
    Args:
        output_dir (str): Directory containing model output folders
    """
    # Load existing results if available
    judge_results = []
    
    # Get all model output directories
    model_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.endswith('_output'):
            model_dirs.append(item)
    
    print(f"Found {len(model_dirs)} model output directories: {model_dirs}")
    
    # Load test instances: key is uid, value is instance
    test_instances = load_test_instances()
    
    for model_name in tqdm(model_dirs, desc="Evaluating models"):
        model_output_dir = os.path.join(output_dir, model_name)
        
        # Load model outputs
        model_outputs = load_model_outputs(model_output_dir)
        
        for instance_id in tqdm(test_instances.keys(), desc=f"Processing {model_name}", leave=False):
            instance_data = test_instances[instance_id]
            
            # Get model output for this instance
            if instance_id not in model_outputs:
                print(f"Warning: No output found for instance {instance_id} in {model_name}")
                continue
                
            model_output = model_outputs[instance_id]
            
            # Parse and load data using the existing parse_and_load_json function
            input_text_list, input_image_list, output_text_list, output_image_list = parse_and_load_json(model_output)
            
            # Prepare input data
            input_images = []
            img_count = -1
            input_img_count = -1
            
            # Load input images
            for i, img_path in enumerate(input_image_list):
                if img_path:
                    full_img_path = os.path.join(OPENING_DIR, img_path)
                    if os.path.exists(full_img_path):
                        try:
                            input_images.append(encode_image(full_img_path))
                            img_count += 1
                            input_img_count += 1
                            input_text_list[i] = input_text_list[i].replace('<BEGIN>','').replace('<image>', '') + f" <IMG_{img_count}>" + f"</IMG_{img_count}>"
                        except Exception as e:
                            print(f"ERROR: {e}")
                            print(full_img_path)
                            input_images.append(None)
                            input_text_list[i] = input_text_list[i].replace('<BEGIN>','').replace('<image>', '')
                    else:
                        input_images.append(None)
                        input_text_list[i] = input_text_list[i].replace('<BEGIN>','').replace('<image>', '')
                else:
                    input_images.append(None)
                    input_text_list[i] = input_text_list[i].replace('<BEGIN>','').replace('<image>', '')
            
            # Process model output
            output_images = []
            
            for i, text in enumerate(output_text_list):
                if output_image_list[i] is not None:
                    img_path = output_image_list[i]
                    full_img_path = os.path.join(model_output_dir, img_path.split('/')[-1])
                    if os.path.exists(full_img_path):
                        try:
                            output_images.append(encode_image(full_img_path))
                            input_img_count += 1
                            output_text_list[i] = text.replace('<image>', '') + f" <IMG_{input_img_count}>" + f"</IMG_{input_img_count}>"
                        except Exception as e:
                            print(f"ERROR: {e}")
                            print(full_img_path)
                            output_text_list[i] = text.replace('<image>', '')
                            output_images.append(None)
                    else:
                        output_text_list[i] = text.replace('<image>', '')
                        output_images.append(None)
                else:
                    output_text_list[i] = text.replace('<image>', '')
                    output_images.append(None)
            
            # Get GPT score
            try:
                score = get_gpt4answer(input_text_list, input_images, output_text_list, output_images)
                print(f"{model_name} - Instance {instance_id}: {score}")
                
                # Save result
                result_data = {
                    'instance_id': instance_id,
                    'model': {'name': model_name.replace('_output', '')},
                    'score': score
                }
                judge_results.append(result_data)
                save_judge_results(judge_results)
                
            except Exception as e:
                print(f"Error evaluating {model_name} for instance {instance_id}: {e}")
                continue

def load_model_outputs(model_output_dir):
    """
    Load model outputs from the specified directory.
    
    Args:
        model_output_dir (str): Path to model output directory
        
    Returns:
        dict: Dictionary of model outputs by instance_id
    """
    model_outputs = {}
    
    # Look for output files in the directory
    output_files = glob.glob(os.path.join(model_output_dir, "*.json"))
    
    for output_file in output_files:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
                instance_uid = os.path.basename(output_file).split('.')[0]

                # Handle different output formats
                model_outputs[instance_uid] = output_data
        except Exception as e:
            print(f"Error loading output file {output_file}: {e}")
            continue
    return model_outputs

def load_test_instances():
    """
    Load test instances from the OpenING dataset.
    
    Returns:
        dict: Dictionary of test instances
    """
    test_instances = {}
    
    # Load test data from OpenING-Benchmark directory
    test_data_path = os.path.join(OPENING_DIR, "test_data.jsonl")
    if os.path.exists(test_data_path):
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    instance = json.loads(line)
                    test_instances[instance['total_uid']] = instance
    else:
        print(f"Warning: Test data file not found at {test_data_path}")
    
    return test_instances


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT Score Evaluation')
    parser.add_argument('--mode', type=str, default='output_dir', 
                       choices=['pairwise_file', 'output_dir'],
                       help='Evaluation mode: pairwise_file or output_dir')
    parser.add_argument('--output_dir', type=str, default='gen_outputs',
                       help='Directory containing model outputs for evaluation')
    
    args = parser.parse_args()

    with open('./prompts/detailed_score_system.txt', 'r') as f:
        file_contents = f.read()
    SYSTEM_MESSAGE = str(file_contents)
    
    if args.mode == 'pairwise_file':
        # Use original logic for pairwise file evaluation
        main_gpt_judge_score_by_pairwise_file_format()
    elif args.mode == 'output_dir':
        if args.output_dir is None:
            print("Error: --output_dir must be specified when using output_dir mode")
            exit()
        # New logic for output directory evaluation
        main_evaluate_output_directory(args.output_dir)
    else:
        print("Invalid mode specified. Use 'pairwise_file' or 'output_dir'")