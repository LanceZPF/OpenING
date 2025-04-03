import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch
import numpy as np
from emu3.mllm.processing_emu3 import Emu3Processor

import json
from tqdm import tqdm

# model path
EMU_HUB = "/mnt/workspace/zpf/.cache/Emu3-Chat"
EMG_HUB = "/mnt/workspace/zpf/.cache/Emu3-Gen"
VQ_HUB = "/mnt/workspace/zpf/.cache/Emu3-VisionTokenizer"

TOTAL_DIR = '/mnt/workspace/zpf/OpenING'
OUTPUT_DIR = './Emu3_output'  # Define your output directory
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

data_path = os.path.join(TOTAL_DIR, "test_data.jsonl")
real_data_list, io_dir_list = load_data(data_path)

print("Data Loaded!")

save_dir = OUTPUT_DIR

i2t_tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
# prepare model and processor
i2t_model = AutoModelForCausalLM.from_pretrained(
    EMU_HUB,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
i2t_model.eval()

image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
i2t_processor = Emu3Processor(image_processor, image_tokenizer, i2t_tokenizer)

I2T_GENERATION_CONFIG = GenerationConfig(pad_token_id=i2t_tokenizer.pad_token_id, bos_token_id=i2t_tokenizer.bos_token_id, eos_token_id=i2t_tokenizer.eos_token_id)
# prepare hyper parameters

# prepare model and processor
t2i_model = AutoModelForCausalLM.from_pretrained(
    EMG_HUB,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
t2i_tokenizer = AutoTokenizer.from_pretrained(EMG_HUB, trust_remote_code=True)

image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True, padding_side="left")
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
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
                image = Image.open(os.path.join(TOTAL_DIR,image_path))
                input_images.append(image)
                temp_ins = input_text_list[img_path_i].replace('<image>', '').replace('<BEGIN>','')
            instruction += input_text_list[img_path_i].replace('<BEGIN>','').replace('<image>', '')
        
        for out_img in out_image_list:
            if out_img != None and len(input_images) < 5:
                input_images.append(out_img)

        for out_text in out_text_list:
            instruction += out_text

        if len(input_images) <=0 or input_images[0] == None:
            # 创建一个224x224的全白图像
            width, height = 224, 224
            white_image = np.full((height, width, 3), 255, dtype=np.uint8)  # 创建一个全白的RGB图像
            # 将numpy数组转换为图像
            image = Image.fromarray(white_image)
            input_images = [image]
        
        inputs = i2t_processor(
                text=instruction,
                image=input_images,
                mode='U',
                padding_image=True,
                padding="longest",
                return_tensors="pt",
            )
        
        outputs = i2t_model.generate(
                inputs.input_ids.to("cuda:0"),
                I2T_GENERATION_CONFIG,
                max_new_tokens=320,
            )
        outputs = outputs[:, inputs.input_ids.shape[-1]:]

        output_text = i2t_processor.batch_decode(outputs, skip_special_tokens=True)[0]

        real_out_text = output_text

        # if len(instruction) > 1000:
        #     prompt = f"The prompt for this generation is: {real_out_text}. The context of this task is: {instruction.replace('[<IMG_PLH>]','')[:500]+instruction.replace('[<IMG_PLH>]','')[-400:]}."
        # else:
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
                unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
            ),
            PrefixConstrainedLogitsProcessor(
                constrained_fn,
                num_beams=1,
            ),
        ])

        # print(pos_inputs.input_ids.shape)

        # generate
        outputs = t2i_model.generate(
            pos_inputs.input_ids.to("cuda:0"),
            T2I_GENERATION_CONFIG,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to("cuda:0"),
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

    save_results(real_data_list[a_index], out_text_list, out_image_path_list)
    torch.cuda.empty_cache()