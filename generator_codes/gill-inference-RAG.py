import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from gill import models
from gill import utils
import time

from tqdm import tqdm
import json
import jsonlines

TOTAL_DIR = '/mnt/workspace/zpf/OpenING'
OUTPUT_DIR = './gill_RAG-dev'  # Define your output directory
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
    data = []  # 初始化数据项列表
    real_data = []

    with open(data_path) as file:  # 打开数据文件
        for line in tqdm(file):  # 遍历每一行
            content = json.loads(line)
            data.append(content)  # 将每行数据加载为JSON对象并添加到列表
            # get input list etc. and return 5 lists
            ainput_list, ainput_image_list, aoutput_list, aoutput_image_list = parse_and_load_json(content)
            real_data.append({"input_text": ainput_list, "input_image": ainput_image_list, "output_text": aoutput_list, "output_image": aoutput_image_list})
    return data, real_data

def save_results(real_data_item, generated_text_list, image_out_list):
    data_uid = real_data_item["total_uid"]
    # Save generated text as JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, f'{data_uid}.jsonl')

    saved_json = real_data_item.copy()
    if 'conversations' in saved_json and len(saved_json['conversations']) > 1:
        saved_json['conversations'][1]['output'] = []

    for index in range(len(generated_text_list)):
        a_out_item = {"text": generated_text_list[index].replace("[IMG1][IMG2][IMG3][IMG4][IMG5][IMG6][IMG7]", "")}
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

def get_image_from_local(path):
    img = Image.open(path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img

def generate_dialogue(prompts: list, system_message: str = None, num_words: int = 32,
                      sf: float = 1.0, temperature: float = 0.0, top_p: float = 1.0,
                      divider_count: int = 40):
    model_dir = 'checkpoints/gill_opt/'
    model = models.load_gill(model_dir)
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)

    start_time = time.time()
    for prompt_idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        out_step = prompt["step"]
        now_step = 0
        out_image_list = []
        out_text_list = []
        
        full_outputs = []

        while now_step < out_step:

            if system_message:
                print("Adding system message")
                full_inputs = [system_message]
            else:
                full_inputs = []
            for p in prompt['input']:
                if p['type'] == 'image':
                    image = get_image_from_local(p['data'])
                    full_inputs.append(image)
                elif p['type'] == 'text' and p['index'] == 0:
                    text = p['data'].replace('<BEGIN> ','').replace(' <image>','')
                    full_inputs.append(f'Q: {text}\n')
                else:
                    text = p['data'].replace('<BEGIN> ','').replace(' <image>','')
                    full_inputs.append(f'{text}\n')

            for p_i in range(min(now_step+1,out_step)):
                p = prompt['output'][p_i]
                if p['type'] == 'image':
                    image = get_image_from_local(p['data'])
                    full_inputs.append(image)
                elif p['type'] == 'text' and p_i == now_step:
                    text = p['data'].replace(' <image>','')
                    full_inputs.append(f'The asnwer should be: {text}. Please repeat the answer.\n')
                else:
                    text = p['data'].replace(' <image>','')
                    full_inputs.append(f'{text}\n')

            # for p in range(len(out_text_list)):
            #     if p < len(out_image_list):
            #         full_inputs.append(f'{out_image_list[p]}\n')
            #     full_inputs.append(f'{out_text_list[p]}\n')
            
            if type(full_inputs[-1]) == str:
                full_inputs[-1] += 'A:'
        
            try:
                return_outputs = model.generate_for_images_and_texts(
                    full_inputs, num_words=num_words, ret_scale_factor=sf,
                    generator=g_cuda, temperature=temperature, top_p=top_p)
            except Exception as e:
                print(e)
                print(full_inputs)
                print(prompt['real_data'])
                now_step += 1
                continue

            for p in return_outputs:
                if type(p) == dict:
                    # Decide whether to retrieve or generate
                    decision_probs = [f'{s:.3f}' for s in p['decision'][1]]
                    if p['decision'][0] == 'gen':
                        out_image = p['gen'][0][0].resize((512, 512))
                        # Generate
                    else:
                        out_image = p['ret'][0][0].resize((512, 512))
                    out_image_list.append(out_image)
                    # full_inputs.append(out_image)
                elif type(p) == str:
                    p_formatted = p.replace('[IMG0] [IMG1] [IMG2] [IMG3] [IMG4] [IMG5] [IMG6] [IMG7]', '')
                    out_text_list.append(p_formatted)
                    # full_inputs.append(p_formatted)
                else:
                    out_image = p
                    out_image_list.append(out_image)
                    # Add outputs
                    # full_inputs.append(out_image)
                now_step += 1
            
            if now_step >= out_step:
                break
        
        save_results(prompt['real_data'], out_text_list, out_image_list)

    # 记录结束时间
    end_time = time.time()
    # 计算推理时间
    total_time = end_time - start_time
    # 使用 divmod 将秒数转换为分钟和秒
    minutes, seconds = divmod(total_time, 60)
    print(f"Inference time: {int(minutes)} minutes and {seconds:.2f} seconds")

    return full_outputs

if __name__ == "__main__":

    saved_id = []
    for file in os.listdir(OUTPUT_DIR):
        if '.jpg' in file and file.split('-')[0] not in saved_id:
            saved_id.append(file.split('-')[0])

    sf = 1.4  # Scaling factor: increase to increase the chance of returning an image
    temperature = 0.0  # 0 means deterministic, try 0.6 for more randomness
    top_p = 1.0  # If you set temperature to 0.6, set this to 0.95
    num_words = 32

    data_path = os.path.join(TOTAL_DIR, 'dev_data.jsonl')
    real_data_list, io_dir_list = load_data(data_path)

    prompts = []

    for a_index, a_data in enumerate(io_dir_list):

        if real_data_list[a_index]['total_uid'] in saved_id:
            continue

        gt_out_step = len(a_data['output_text'])
        out_image_list = []
        out_text_list = []
        gtout_text_list = a_data['output_text']
        gtout_image_list = a_data['output_image']

        a_prompt = {"step": gt_out_step, "input": [], "output": []}

        for index in range(len(a_data['input_text'])):
            input_image_path = a_data['input_image'][index]
            if input_image_path:
                a_prompt["input"].append({"type": "image", "data": os.path.join(TOTAL_DIR,input_image_path), "index": index})

            a_prompt["input"].append({"type": "text", "data": a_data['input_text'][index], "index": index})

        for index in range(len(gtout_text_list)):
            if index < len(gtout_image_list) and gtout_image_list[index]:
                a_prompt["output"].append({"type": "image", "data": os.path.join(TOTAL_DIR,gtout_image_list[index]), "index": index})
            a_prompt['output'].append({"type": "text", "data": gtout_text_list[index], "index": index})

        a_prompt['real_data'] = real_data_list[a_index]
        prompts.append(a_prompt)

    full_outputs = generate_dialogue(prompts, num_words=num_words, sf=sf, temperature=temperature, top_p=top_p)