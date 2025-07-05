import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
import os
import torch
import torch.multiprocessing as mp
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import numpy as np
from emu3.mllm.processing_emu3 import Emu3Processor
import json
from tqdm import tqdm
import re

# Model paths
EMU_HUB = "/mnt/petrelfs/zhoupengfei/zpf/.cache/Emu3-Chat"
EMG_HUB = "/mnt/petrelfs/zhoupengfei/zpf/.cache/Emu3-Gen"
VQ_HUB = "/mnt/petrelfs/zhoupengfei/.cache/Emu3-VisionTokenizer"

TOTAL_DIR = '/mnt/petrelfs/zhoupengfei/zpf/OpenING'
INPUT_DIR = os.path.join(TOTAL_DIR, 'InputImages')
OUTPUT_DIR = './Emu3_output'  # Define your output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

pattern = r"/(\d+)-([io]-\d+\.jpg)$"
file_path = "/mnt/petrelfs/zhoupengfei/zpf/OpenING/test_data.jsonl"

def generate_total_uid(meta_task_id, subtask_id, data_id):
    # Ensure meta_task_id is 2 digits, subtask_id is 2 digits, and data_id is 3 digits
    return f'{int(meta_task_id):02}{int(subtask_id):02}{int(data_id):03}'
image_to_uid = {}
# Read the JSONL file line by line
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
            io_data.append({"input_text": ainput_list, "input_image": ainput_image_list, "output_text": aoutput_list, "output_image": aoutput_image_list})
    
    return ori_data, io_data

def save_worker_results(real_data_item, generated_text_list, image_out_list):
    data_uid = real_data_item["total_uid"]
    jsonl_path = os.path.join(OUTPUT_DIR, f'{data_uid}.jsonl')

    saved_json = real_data_item.copy()
    if 'conversations' in saved_json and len(saved_json['conversations']) > 1:
        saved_json['conversations'][1]['output'] = []

    for index in range(max(len(generated_text_list), len(image_out_list))):
        a_out_item = {"text": generated_text_list[index].strip() if index < len(generated_text_list) else ""}
        a_out_item["image"] = image_out_list[index] if index < len(image_out_list) else None
        saved_json['conversations'][1]['output'].append(a_out_item)

    with open(jsonl_path, mode='w', encoding='utf-8') as writer:
        json.dump(saved_json, writer, indent=4)

def worker(gpu_id, io_dir_list, real_data_list):
    torch.cuda.set_device(gpu_id)
    
    i2t_tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
    i2t_model = AutoModelForCausalLM.from_pretrained(
        EMU_HUB,
        device_map=f"cuda:{gpu_id}",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).eval()

    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map=f"cuda:{gpu_id}", trust_remote_code=True).eval()
    i2t_processor = Emu3Processor(image_processor, image_tokenizer, i2t_tokenizer)

    I2T_GENERATION_CONFIG = GenerationConfig(pad_token_id=i2t_tokenizer.pad_token_id, bos_token_id=i2t_tokenizer.bos_token_id, eos_token_id=i2t_tokenizer.eos_token_id)

    # Prepare second model
    t2i_model = AutoModelForCausalLM.from_pretrained(
        EMG_HUB,
        device_map=f"cuda:{gpu_id}",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    t2i_tokenizer = AutoTokenizer.from_pretrained(EMG_HUB, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True, padding_side="left")
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map=f"cuda:{gpu_id}", trust_remote_code=True).eval()
    t2i_processor = Emu3Processor(image_processor, image_tokenizer, t2i_tokenizer)
    
    POSITIVE_PROMPT = " masterpiece, film grained, best quality."
    NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
    classifier_free_guidance = 3.0

    T2I_GENERATION_CONFIG = GenerationConfig(
        use_cache=True,
        eos_token_id=t2i_model.config.eos_token_id,
        pad_token_id=t2i_model.config.pad_token_id,
        max_new_tokens=40960,
        do_sample=True,
        top_k=2048,
    )

    for a_index, a_data in enumerate(tqdm(io_dir_list)):

        input_image_path = a_data['input_image']
        input_text_list = a_data['input_text']
        gt_out_step = len(a_data['output_text'])
        a_real_data = real_data_list[a_index]
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
                    image = image_path
                    img_tem = image_path
                    temp_uid = image_to_uid[img_tem]
                    new_image_path = re.sub(pattern, rf"/{temp_uid}-\2", image)
                    new_img_name = new_image_path.split('/')[-1]
                    image_path = os.path.join(INPUT_DIR, new_img_name)
                    
                    # image_path = os.path.join(INPUT_DIR, temp_img_path.replace(f'{original_id}-i-',f'{total_uid}-i-'))

                    image = Image.open(image_path)
                    image = image.resize((256, 256))  # Resize image to 256x256
                    input_images.append(image)
                    # temp_ins = input_text_list[img_path_i].replace('<image>', '').replace('<BEGIN>','')
                instruction += input_text_list[img_path_i].replace('<BEGIN>','').replace('<image>', '')
            
            for out_img in out_image_list:
                if out_img != None and len(input_images) < 5:
                    if isinstance(out_img, Image.Image):
                        out_img = out_img.resize((256, 256))  # Resize output image to 256x256
                    input_images.append(out_img)

            for out_text in out_text_list:
                instruction += out_text

            if len(input_images) <=0 or input_images[0] == None:
                # Create a 256x256 white image instead of 224x224
                width, height = 256, 256
                white_image = np.full((height, width, 3), 255, dtype=np.uint8)
                image = Image.fromarray(white_image)
                input_images = [image]
            try:
                inputs = i2t_processor(
                        text=instruction,
                        image=input_images[-1],
                        mode='U',
                        padding_image=True,
                        padding="longest",
                        return_tensors="pt",
                    )
            except Exception as e:
                print(e)
                break
            
            inputs = {k: v.to(f"cuda:{gpu_id}") for k, v in inputs.items()}

            # Use the dictionary keys to access `input_ids`
            outputs = i2t_model.generate(
                inputs["input_ids"],
                I2T_GENERATION_CONFIG,
                max_new_tokens=320,
            )
            outputs = outputs[:, inputs["input_ids"].shape[-1]:]

            try:
                output_text = i2t_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            except Exception as e:
                print(e)
                break

            real_out_text = output_text

            prompt = [f"The prompt for this generation is: {real_out_text}. The context of this task is: {instruction.replace('[<IMG_PLH>]','')}" + POSITIVE_PROMPT]

            kwargs = dict(
                mode='G',
                ratio=["1:1"],
                image_area=t2i_model.config.image_area,
                return_tensors="pt",
                padding="longest",
            )
            pos_inputs = t2i_processor(text=prompt, **kwargs)
            neg_inputs = t2i_processor(text=[NEGATIVE_PROMPT], **kwargs)

            h, w = pos_inputs.image_size[0]

            constrained_fn = t2i_processor.build_prefix_constrained_fn(h, w)
            logits_processor = LogitsProcessorList([
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    classifier_free_guidance,
                    t2i_model,
                    unconditional_ids=neg_inputs.input_ids.to(f"cuda:{gpu_id}"),
                ),
                PrefixConstrainedLogitsProcessor(
                    constrained_fn,
                    num_beams=1,
                ),
            ])

            outputs = t2i_model.generate(
                pos_inputs.input_ids.to(f"cuda:{gpu_id}"),
                T2I_GENERATION_CONFIG,
                logits_processor=logits_processor,
                attention_mask=pos_inputs.attention_mask.to(f"cuda:{gpu_id}"),
            )

            mm_list = t2i_processor.decode(outputs[0])

            count_i_num = 0
            for idx, im in enumerate(mm_list):
                if not isinstance(im, Image.Image):
                    continue
                save_path = os.path.join(OUTPUT_DIR, f'{real_data_list[a_index]["total_uid"]}-o-{i_step}.jpg')
                im.save(save_path)
                out_image_path_list.append(save_path)
                i_step += 1
                count_i_num += 1
            
            if count_i_num > 1:
                # split the real_out_text into certain count_i_num parts
                real_out_text_list = real_out_text.split('.')
                if len(real_data_list) >= count_i_num:
                    for i in range(count_i_num-1):
                        out_text_list.append(real_out_text_list[i] + '.')
                    # append the rest as a single string
                    out_text_list.append(''.join(real_out_text_list[count_i_num-1:]))
                else:
                    for i in range(len(real_data_list)):
                        out_text_list.append(real_out_text_list[i] + '.')
                    for i in range(len(real_data_list), count_i_num):
                        out_text_list.append("<More visualized images here>")
            else:
                out_text_list.append(real_out_text)
                
        save_worker_results(real_data_list[a_index], out_text_list, out_image_list)
        torch.cuda.empty_cache()

def merge_worker_outputs(worker_output_files, merged_output_file):
    with open(merged_output_file, mode='w', encoding='utf-8') as merged_writer:
        for output_file in worker_output_files:
            with open(output_file, mode='r', encoding='utf-8') as worker_file:
                for line in worker_file:
                    merged_writer.write(line)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    data_path = os.path.join(TOTAL_DIR, "test_data.jsonl")
    real_data_list, io_dir_list = load_data(data_path)

    # Determine saved IDs to avoid reprocessing
    saved_id = []
    for file in os.listdir(OUTPUT_DIR):
        if '.jpg' in file and file.split('-')[0] not in saved_id:
            saved_id.append(file.split('-')[0])

    # Filter out the already processed data
    filtered_real_data_list = []
    filtered_io_dir_list = []

    count = 0
    for real_data, io_data in zip(real_data_list, io_dir_list):
        if real_data["total_uid"] not in saved_id:
        # if real_data["total_uid"] not in saved_id and count >= len(real_data_list)/2:
            filtered_real_data_list.append(real_data)
            filtered_io_dir_list.append(io_data)
        count += 1

    # Update real_data_list and io_dir_list with filtered data
    real_data_list = filtered_real_data_list
    io_dir_list = filtered_io_dir_list
    
    # Divide data among available GPUs
    num_gpus = torch.cuda.device_count()
    data_chunks = len(io_dir_list) // num_gpus

    # Launch worker processes
    processes = []
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * data_chunks
        end_idx = (gpu_id + 1) * data_chunks if gpu_id != num_gpus - 1 else len(io_dir_list)
        p = mp.Process(target=worker, args=(gpu_id, io_dir_list[start_idx:end_idx], real_data_list[start_idx:end_idx]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        p.join()