import os
import shutil
import argparse
import random
import time
import json
from tqdm import tqdm
import re
import base64
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import hydra
import pyrootutils
import os
import torch
from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import numpy as np

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
HYDRA_FULL_ERROR=1

os.system('clear')

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

TOTAL_DIR = ""
data_type = 'test'
io_idx = 0
OUTPUT_DIR = 'seed_llama14b_output/%s/'%(data_type)  # Define your output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate(tokenizer, input_tokens, generation_config, model, device = 'cuda:0'):

    input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
    input_ids = input_ids.to(device)

    generate_ids = model.generate(
        input_ids=input_ids,
        **generation_config
    )
    generate_ids = generate_ids[0][input_ids.shape[1]:]
    
    return generate_ids

def decode_image_text(generate_ids, tokenizer, gen_image = False):

    boi_list = torch.where(generate_ids == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]

    if gen_image==False or len(boi_list) == 0 or  len(eoi_list) == 0:
        text_ids = generate_ids
        texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        images = None
    else:
        boi_index = boi_list[0]
        eoi_index = eoi_list[0]

        text_ids = generate_ids[:boi_index]
        if len(text_ids) != 0:
            texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        else:
            texts = None
            
        image_ids = (generate_ids[boi_index+1:eoi_index] - image_id_shift).reshape(1,-1)

        images = tokenizer.decode_image(image_ids)
    return texts, images



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

    with open(data_path) as file:  # 打开数据文件
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
                image_out_list[index][0].save(image_path)
            else:
                a_out_item["image"] = None
        else:
            a_out_item["image"] = None

        saved_json['conversations'][1]['output'].append(a_out_item)

    with open(jsonl_path, mode='w', encoding='utf-8') as writer:
        json.dump(saved_json, writer, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-path', type=str, default=TOTAL_DIR, help="Folder where the CSV and the images were downloaded.")
    parser.add_argument('--data-file-name', type=str, default="%s_data.jsonl" % data_type, help="Folder where the CSV and the images were downloaded.")
    args = parser.parse_args()
    data_path = os.path.join(args.meta_path, args.data_file_name)
    
    saved_id = []
    for file in os.listdir(OUTPUT_DIR):
        if '.jsonl' in file and file.split('.')[0] not in saved_id:
            saved_id.append(file.split('.')[0])

    real_data_list, io_dir_list = load_data(data_path)

    device = "cuda:%d"%io_idx

    tokenizer_cfg_path = 'configs/tokenizer/seed_llama_tokenizer_hf.yaml'
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)

    transform_cfg_path = 'configs/transform/clip_transform.yaml'
    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    model_cfg = OmegaConf.load('configs/llm/seed_llama_14b.yaml')
    model0 = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
    model0 = model0.eval().to(device)

    generation_config = {
            'temperature': 1.0,
            'num_beams': 1,
            'max_new_tokens': 512,
            'top_p': 0.5,
            'do_sample': True
        }

    s_token = "USER:"
    e_token = "ASSISTANT:"
    sep = "\n"
    
    a_index_start = 0

    for a_index in range(a_index_start, len(io_dir_list)):
        model = model0
        
        a_data = io_dir_list[a_index]
        input_image_path = a_data['input_image']
        input_text_list  = a_data['input_text']
        num_out_step = len(a_data['output_text'])
        data_uid = real_data_list[a_index]['total_uid']
    
        if real_data_list[a_index]['total_uid'] in saved_id:
            continue
 
        subtask_name = real_data_list[a_index]['subtask_name']
        meta_task_name = real_data_list[a_index]['meta_task_name']
        data_id = real_data_list[a_index]['data_id']
        
        prompt = input_text_list[0]
        prompt = prompt.replace('<BEGIN>', '')
        prompt = prompt.replace('<image>', '')
        num_input_images = len(input_image_path)

        print('- IDX[%d], UID[%s], DataID[%s]: Task [%s], Subtask [%s], Input Images [%d]' % (a_index, data_uid, data_id, meta_task_name, subtask_name, num_input_images))
        
        img_token_list= ''
        for i in range(num_input_images):
            fpath = input_image_path[i]
            if fpath is not None:
                image = Image.open(input_image_path[i]).convert('RGB')
                image_tensor = transform(image).to(device)
                img_ids = tokenizer.encode_image(image_torch=image_tensor)
                img_ids = img_ids.view(-1).cpu().numpy()
                img_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in img_ids]) + EOI_TOKEN
                img_token_list+=img_tokens

        image_out_list = []
        generated_text_list =[]
        if num_out_step == 1:
            input_tokens = tokenizer.bos_token + s_token + 'Based on the input %d image, ' % num_input_images + img_token_list
            input_tokens +=  prompt +'. Summarize your answer in one sentence. '+ e_token + sep
            generate_ids = generate(tokenizer, input_tokens, generation_config, model, device = device)
            answer0, _ = decode_image_text(generate_ids, tokenizer)
            generated_text_list.append(answer0)
            print('- ')
            print(input_tokens)
            print('- ')
            print(answer0)
            
            img_gen_prompt = 'Please generatea an image according to: ' + answer0
            input_tokens = tokenizer.bos_token + s_token + img_gen_prompt + e_token + sep
            generate_ids = generate(tokenizer, input_tokens, generation_config, model, device = device)
            _, image = decode_image_text(generate_ids, tokenizer, gen_image = True)
            image_out_list.append(image)

        else:
            input_tokens = tokenizer.bos_token + s_token + 'Based on the input %d images, ' % num_input_images + img_token_list
            input_tokens +=  prompt + 'Describe in %d steps. ' % num_out_step + e_token + sep
            print('- ')
            print(input_tokens)
            generate_ids = generate(tokenizer, input_tokens, generation_config, model, device = device)

            answer0, _ = decode_image_text(generate_ids, tokenizer)
            print('- ')
            #print(answer0)
            answer_split = answer0.split('\n')

            for i in range(np.min([len(answer_split), num_out_step])):
                if answer_split[i] !='' and ('Here is ' not in answer_split[i]):
                    generated_text_list.append(answer_split[i])
                    print(answer_split[i])
                    img_gen_prompt = "Could you generate an image according to the description of step %d: %s"%(i, answer_split[i])
                    input_tokens = tokenizer.bos_token + s_token + img_gen_prompt + e_token + sep
                    generate_ids = generate(tokenizer, input_tokens, generation_config, model, device = device)
                    _, image = decode_image_text(generate_ids, tokenizer, gen_image = True)
                    image_out_list.append(image)

        save_results(real_data_list[a_index], generated_text_list, image_out_list)
        print('\n')
        
