import os
import torch
import torch.multiprocessing as mp
from PIL import Image
import numpy as np
import vila_u
import json
from tqdm import tqdm
import re
import cv2

# Define directories
TOTAL_DIR = '/mnt/petrelfs/zhoupengfei/zpf/OpenING'
INPUT_DIR = os.path.join(TOTAL_DIR, 'InputImages')
OUTPUT_DIR = './VILA-U_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

pattern = r"/(\d+)-([io]-\d+\.jpg)$"
file_path = os.path.join(TOTAL_DIR, "test_data.jsonl")

def generate_total_uid(meta_task_id, subtask_id, data_id):
    return f'{int(meta_task_id):02}{int(subtask_id):02}{int(data_id):03}'

# Build image to UID mapping
image_to_uid = {}
with open(file_path, "r") as file:
    for line in file:
        data = json.loads(line.strip())
        conversations = data.get("conversations", [])
        for conversation in conversations:
            for output in conversation.get("input", []):
                image_path = output.get("image")
                if image_path:
                    match = re.search(pattern, image_path)
                    data_id = data['data_id']
                    if match:
                        data_id = match.group(1)
                    total_uid = generate_total_uid(data['meta_task_id'], 
                                                data['subtask_id'], 
                                                data_id)
                    image_to_uid[image_path] = total_uid

def parse_and_load_json(content):
    input_text_list = []
    input_image_list = []
    output_text_list = []
    output_image_list = []

    for input_content in content['conversations'][0]['input']:
        input_text_list.append(input_content['text'].strip())
        input_image_list.append(input_content['image'])

    for output_content in content['conversations'][1]['output']:
        output_text_list.append(output_content['text'].strip())
        output_image_list.append(output_content['image'])

    return input_text_list, input_image_list, output_text_list, output_image_list

def load_data(data_path):
    ori_data = []
    io_data = []

    with open(data_path) as file:
        for line in tqdm(file):
            content = json.loads(line)
            ori_data.append(content)
            ainput_list, ainput_image_list, aoutput_list, aoutput_image_list = parse_and_load_json(content)
            io_data.append({
                "input_text": ainput_list,
                "input_image": ainput_image_list,
                "output_text": aoutput_list,
                "output_image": aoutput_image_list
            })
    
    return ori_data, io_data

def save_worker_results(real_data_item, generated_text_list, image_out_list):
    data_uid = real_data_item["total_uid"]
    jsonl_path = os.path.join(OUTPUT_DIR, f'{data_uid}.jsonl')

    saved_json = real_data_item.copy()
    if 'conversations' in saved_json and len(saved_json['conversations']) > 1:
        saved_json['conversations'][1]['output'] = []

    for index in range(max(len(generated_text_list), len(image_out_list))):
        a_out_item = {
            "text": generated_text_list[index].strip() if index < len(generated_text_list) else "",
            "image": image_out_list[index] if index < len(image_out_list) else None
        }
        saved_json['conversations'][1]['output'].append(a_out_item)

    with open(jsonl_path, mode='w', encoding='utf-8') as writer:
        json.dump(saved_json, writer, indent=4)

def save_image(response, save_path):
    for i in range(response.shape[0]):
        image = response[i].permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image)

def worker(gpu_id, io_dir_list, real_data_list, model_path, cfg, generation_nums):
    torch.cuda.set_device(gpu_id)
    
    # Load Vila-U model
    model = vila_u.load(model_path)
    model.to(f"cuda:{gpu_id}")
    
    for a_index, a_data in enumerate(tqdm(io_dir_list)):
        input_image_path = a_data['input_image']
        input_text_list = a_data['input_text']
        gt_out_step = len(a_data['output_text'])
        a_real_data = real_data_list[a_index]
        out_image_path_list = []
        out_text_list = []

        i_step = 0
        while i_step < gt_out_step:
            # Process input images
            current_images = []
            instruction = ''
            
            if input_image_path[0] == None and i_step == 0:
                try:
                    for text_i in range(len(input_text_list)):
                        instruction += input_text_list[text_i].replace('<BEGIN>','').replace('<image>', '')
                    output_image1 = model.generate_image_content(instruction, cfg, generation_nums)
                    save_path = os.path.join(OUTPUT_DIR, f'{a_real_data["total_uid"]}-o-{i_step}.jpg')
                    save_image(output_image1, save_path)
                    out_image_path_list.append(save_path)
                    i_step+=1
                    loaded_image = vila_u.Image(save_path)
                    current_images.append(loaded_image)
                    output_text1 = model.generate_content([loaded_image, instruction])
                    temp_text_list = output_text1.split('\n')
                    temp_text_list = [text.strip() for text in temp_text_list if text.strip() and text.strip() != '\n' and len(text.strip()) >= 2]
                    
                    if len(temp_text_list) > gt_out_step:
                        # Calculate how many segments to combine
                        segments_per_output = len(temp_text_list) // gt_out_step
                        remainder = len(temp_text_list) % gt_out_step
                        
                        split_text_list = []
                        start_idx = 0
                        
                        for i in range(gt_out_step):
                            extra = 1 if i < remainder else 0
                            end_idx = start_idx + segments_per_output + extra
                            combined_text = '\n'.join(temp_text_list[start_idx:end_idx])
                            split_text_list.append(combined_text)
                            start_idx = end_idx
                    else:
                        split_text_list = temp_text_list
                    
                    out_text_list.extend(split_text_list)
                    for text_i in range(1,len(out_text_list)):
                        temp_instruction = out_text_list[text_i] + instruction
                        instruction += out_text_list[text_i]
                        img_response = model.generate_image_content(temp_instruction, cfg, generation_nums)
                        save_path = os.path.join(OUTPUT_DIR, f'{a_real_data["total_uid"]}-o-{i_step}.jpg')
                        save_image(img_response, save_path)
                        out_image_path_list.append(save_path)
                        i_step += 1
                    if i_step >= gt_out_step:
                        break
                    
                except Exception as e:
                    print(f"1Error processing step {i_step}: {e}")
                    break
                
            for img_path_i in range(len(input_image_path)):
                image_path = input_image_path[img_path_i]
                if image_path:
                    img_tem = image_path
                    temp_uid = image_to_uid[img_tem]
                    new_image_path = re.sub(pattern, rf"/{temp_uid}-\2", image_path)
                    new_img_name = new_image_path.split('/')[-1]
                    image_path = os.path.join(INPUT_DIR, new_img_name)
                    # Load image using Vila-U's image loader
                    image = vila_u.Image(image_path)
                    current_images.append(image)
                    instruction += input_text_list[img_path_i].replace('<BEGIN>','').replace('<image>', '')

            # Add previous output images to context if any
            # for out_img in out_image_list:
            #     if out_img is not None:
            #         current_images.append(vila_u.Image(out_img))

            # Generate response
            try:
                # Split text into segments if needed
                temp_text_list = []
                inputs = [current_images[0], instruction] if current_images else instruction
                response = model.generate_content(inputs)
                temp_text_list = response.split('\n')
                temp_text_list = [text.strip() for text in temp_text_list if text.strip() and text.strip() != '\n' and len(text.strip()) >= 2]
                
                now_out_step = gt_out_step - len(out_text_list)

                if len(temp_text_list) > now_out_step:
                    # Calculate segments per output and remainder
                    segments_per_output = len(temp_text_list) // now_out_step
                    remainder = len(temp_text_list) % now_out_step
                    
                    split_text_list = []
                    start_idx = 0
                    
                    for i in range(now_out_step):
                        # Add one extra segment if there are remaining segments
                        extra = 1 if i < remainder else 0
                        end_idx = start_idx + segments_per_output + extra
                        
                        # Join segments with newline
                        combined_text = '\n'.join(temp_text_list[start_idx:end_idx])
                        split_text_list.append(combined_text)
                        
                        start_idx = end_idx
                else:
                    split_text_list = temp_text_list

                out_text_list.extend(split_text_list)
                
                for text_i in range(len(split_text_list)):
                    temp_instruction = out_text_list[text_i] + instruction
                    instruction += out_text_list[text_i]
                    img_response = model.generate_image_content(temp_instruction, cfg, generation_nums)
                    save_path = os.path.join(OUTPUT_DIR, f'{a_real_data["total_uid"]}-o-{i_step}.jpg')
                    save_image(img_response, save_path)
                    out_image_path_list.append(save_path)
                    i_step += 1
                if i_step >= gt_out_step:
                    break

            except Exception as e:
                print(f"2Error processing step {i_step}: {e}")
                break
                
        save_worker_results(real_data_list[a_index], out_text_list, out_image_path_list)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/petrelfs/zhoupengfei/zpf/.cache/vila-u-7b-256")
    parser.add_argument("--data_path", type=str, default=file_path)
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.9, help="The value of temperature for text generation.")
    parser.add_argument("--top_p", type=float, default=0.6, help="The value of top-p for text generation.")
    ### image and video generation arguments
    parser.add_argument("--cfg", type=float, default=3.0, help="The value of the classifier free guidance for image generation.")
    parser.add_argument("--generation_nums", type=int, default=1)
    args = parser.parse_args()

    mp.set_start_method("spawn")
    real_data_list, io_dir_list = load_data(args.data_path)

    # Filter already processed data
    saved_id = []
    for file in os.listdir(args.output_dir):
        if '.jpg' in file and file.split('-')[0] not in saved_id:
            saved_id.append(file.split('-')[0])

    filtered_real_data_list = []
    filtered_io_dir_list = []
    for real_data, io_data in zip(real_data_list, io_dir_list):
        if real_data["total_uid"] not in saved_id:
            filtered_real_data_list.append(real_data)
            filtered_io_dir_list.append(io_data)

    real_data_list = filtered_real_data_list
    io_dir_list = filtered_io_dir_list

    # Multi-GPU processing
    num_gpus = torch.cuda.device_count()
    data_chunks = len(io_dir_list) // num_gpus

    processes = []
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * data_chunks
        end_idx = (gpu_id + 1) * data_chunks if gpu_id != num_gpus - 1 else len(io_dir_list)
        p = mp.Process(
            target=worker, 
            args=(gpu_id, 
                  io_dir_list[start_idx:end_idx], 
                  real_data_list[start_idx:end_idx],
                  args.model_path,args.cfg,args.generation_nums)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()