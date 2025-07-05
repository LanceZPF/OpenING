import requests
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
from PIL import Image
import re
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import cv2
from diffusers import DiffusionPipeline
import numpy as np

import json

from tqdm import tqdm

TOTAL_DIR = '/mnt/workspace/zpf/OpenING'
OUTPUT_DIR = './Emu2-q_dumb-dev'  # Define your output directory
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

    for index in range(max(len(generated_text_list),len(image_out_list))):
        if index < len(generated_text_list):
            a_out_item = {"text": generated_text_list[index].strip()}
        else:
            a_out_item = {"text": ""}
        if index < len(image_out_list):
            if image_out_list[index] is not None:
                a_out_item["image"] = image_out_list[index]
                # image_path = os.path.join(OUTPUT_DIR, f'{data_uid}-o-{index}.jpg')
                # image_out_list[index].save(image_path)
            else:
                a_out_item["image"] = None
        else:
            a_out_item["image"] = None

        saved_json['conversations'][1]['output'].append(a_out_item)

    with open(jsonl_path, mode='w', encoding='utf-8') as writer:
        json.dump(saved_json, writer, indent=4)

saved_id = []
for file in os.listdir(OUTPUT_DIR):
    if '.jpg' in file and file.split('-')[0] not in saved_id:
        saved_id.append(file.split('-')[0])

data_path = os.path.join(TOTAL_DIR, "dev_data.jsonl")
real_data_list, io_dir_list = load_data(data_path)

print("Data Loaded!")

save_dir = OUTPUT_DIR

i2t_load_path = '/mnt/workspace/zpf/.cache/Emu2-Chat'
i2t_tokenizer = AutoTokenizer.from_pretrained(i2t_load_path) # "BAAI/Emu2-Chat"
# i2t_model = AutoModelForCausalLM.from_pretrained(
#     i2t_load_path, # "BAAI/Emu2-Chat"
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True).to('cuda:0').eval()

i2t_model = AutoModelForCausalLM.from_pretrained(
    i2t_load_path, # "BAAI/Emu2-Chat"
    load_in_4bit=True,
    trust_remote_code=True, 
    bnb_4bit_compute_dtype=torch.float16).eval()

# multi-gpu
# with init_empty_weights():
#     i2t_model = AutoModelForCausalLM.from_pretrained(
#         i2t_load_path, # "BAAI/Emu2-Chat"
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True)
# device_map = infer_auto_device_map(i2t_model, max_memory={0:'38GiB',1:'38GiB'}, no_split_module_classes=['Block','LlamaDecoderLayer'])
# device_map["model.decoder.lm.lm_head"] = 0
# i2t_model = load_checkpoint_and_dispatch(
#     i2t_model, 
#     i2t_load_path,
#     device_map=device_map).eval()

t2i_load_path = "/mnt/workspace/zpf/.cache/Emu2-Gen"
multimodal_encoder = AutoModelForCausalLM.from_pretrained(
    f"{t2i_load_path}/multimodal_encoder",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16"
)
t2i_tokenizer = AutoTokenizer.from_pretrained(f"{t2i_load_path}/tokenizer")
t2i_pipe = DiffusionPipeline.from_pretrained(
    t2i_load_path,
    custom_pipeline="pipeline_emu2_gen",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16",
    multimodal_encoder=multimodal_encoder,
    tokenizer=t2i_tokenizer,
)
t2i_pipe.to("cuda:1")

print('Image token preparation done')

for a_index, a_data in enumerate(tqdm(io_dir_list)):

    if real_data_list[a_index]['total_uid'] in saved_id:
        continue

    input_image_path = a_data['input_image']
    input_text_list = a_data['input_text']
    gt_out_step = len(a_data['output_text'])
    out_image_list = []
    out_image_path_list = []
    out_text_list = []

    i_step = 0
    while i_step < gt_out_step:
        input_images = []
        
        instruction = ''
        
        for img_path_i in range(len(input_image_path)):
            image_path = input_image_path[img_path_i]
            if image_path:
                if len(input_images) < 5:
                    image = Image.open(os.path.join(TOTAL_DIR,image_path)).convert('RGB')
                    input_images.append(image)
                    temp_ins = input_text_list[img_path_i].replace('<image>', '').replace('<BEGIN>','')
                    instruction = instruction + '[<IMG_PLH>]' + temp_ins
                elif input_image_path[0]:
                    image = Image.open(os.path.join(TOTAL_DIR,input_image_path[0])).convert('RGB')
                    input_images.append(image)
                    temp_ins = input_text_list[img_path_i].replace('<image>', '').replace('<BEGIN>','')
                    instruction = instruction + '[<IMG_PLH>]' + temp_ins
                    break
            else:
                instruction += input_text_list[img_path_i].replace('<BEGIN>','').replace('<image>', '')
        
        # for out_img in out_image_list:
        #     if out_img != None and len(input_images) < 5:
        #         input_images.append(out_img)

        for out_text in out_text_list:
        #     if '[<IMG_PLH>]' not in out_text:
        #         out_text = '[<IMG_PLH>]' + out_text
            instruction += out_text
        
        print(instruction)
        if len(input_images) > 0:
            inputs = i2t_model.build_input_ids(
                text=[instruction],
                tokenizer=i2t_tokenizer,
                image=input_images
            )
        else:
            inputs = i2t_model.build_input_ids(
                text=[instruction],
                tokenizer=i2t_tokenizer
            )
        try:
            with torch.no_grad():
                if len(input_images) > 0:
                    outputs = i2t_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        image=inputs["image"].to(torch.float16),
                        max_new_tokens=64,
                        length_penalty=-1)
                else:
                    outputs = i2t_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        image=None, # should be torch.bfloat16
                        max_new_tokens=64,
                        length_penalty=-1)
        except Exception as e:
            print(e)
            break
        output_text = i2t_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        real_out_text = ""
        for out_text in output_text:
            real_out_text += out_text.strip()
        out_text_list.append(real_out_text)

        if len(instruction) > 1000:
            prompt = f"The prompt for this generation is: {real_out_text}. The context of this task is: {instruction.replace('[<IMG_PLH>]','')[:500]+instruction.replace('[<IMG_PLH>]','')[-400:]}."
        else:
            prompt = f"The prompt for this generation is: {real_out_text}. The context of this task is: {instruction.replace('[<IMG_PLH>]','')}."

        if len(input_images) > 0:
            gen_input = []
            # for i_img in input_images:
            if input_images[0] != None:
                gen_input.append(input_images[0])
            gen_input.append(prompt)
        else:
            gen_input = prompt

        with torch.no_grad():
            print(gen_input)
            ret = t2i_pipe(gen_input)
        out_a_img = ret.image
        save_path = os.path.join(OUTPUT_DIR, f'{real_data_list[a_index]["total_uid"]}-o-{i_step}.jpg')
        out_a_img.save(save_path)
        out_image_list.append(out_a_img)
            
        out_image_path_list.append(save_path)
            
        i_step += 1
    save_results(real_data_list[a_index], out_text_list, out_image_path_list)
    torch.cuda.empty_cache()