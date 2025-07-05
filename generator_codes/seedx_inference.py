import hydra
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pyrootutils
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from any_res import process_anyres_image
import re
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

import json

from tqdm import tqdm

TOTAL_DIR = '/mnt/workspace/zpf/OpenING'
OUTPUT_DIR = './seedx_dumb-dev'  # Define your output directory
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

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
BOP_TOKEN = '<patch>'
EOP_TOKEN = '</patch>'
IMG_TOKEN = '<img_{:05d}>'
resolution_grids = ['1x1']
base_resolution = 448

device = 'cuda'
dtype = torch.float16
dtype_str = 'fp16'
num_img_in_tokens = 64
num_img_out_tokens = 64

instruction_prompt = '[INST] {instruction} [/INST]\n'
generation_prompt = '[INST] Generate an image: {instruction} [/INST]\n'

tokenizer_cfg_path = 'configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml'
image_transform_cfg_path = 'configs/processer/qwen_448_transform.yaml'
visual_encoder_cfg_path = 'configs/visual_encoder/qwen_vitg_448.yaml'
llm_cfg_path = 'configs/clm_models/llm_seed_x_i.yaml'
agent_cfg_path = 'configs/clm_models/agent_seed_x_i.yaml'
adapter_cfg_path = 'configs/sdxl_adapter/sdxl_qwen_vit_resampler_l4_q64_pretrain_no_normalize.yaml'
discrete_model_cfg_path = 'configs/discrete_model/discrete_identity.yaml'

diffusion_model_path = 'pretrained/stable-diffusion-xl-base-1.0'

save_dir = OUTPUT_DIR

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg)

image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
image_transform = hydra.utils.instantiate(image_transform_cfg)

visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
visual_encoder.eval().to(device, dtype=dtype)
print('Init visual encoder done')

llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype)
print('Init llm done.')

agent_model_cfg = OmegaConf.load(agent_cfg_path)
agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)

agent_model.eval().to(device, dtype=dtype)
print('Init agent mdoel Done')

noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
print('init vae')
vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device, dtype=dtype)
print('init unet')
unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(device, dtype=dtype)

adapter_cfg = OmegaConf.load(adapter_cfg_path)
adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device, dtype=dtype).eval()

discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(device).eval()
print('Init adapter done')

adapter.init_pipe(vae=vae,
                  scheduler=noise_scheduler,
                  visual_encoder=visual_encoder,
                  image_transform=image_transform,
                  discrete_model=discrete_model,
                  dtype=dtype,
                  device=device)

print('Init adapter pipe done')

boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]

grid_pinpoints = []
for scale in resolution_grids:
    s1, s2 = scale.split('x')
    grid_pinpoints.append([int(s1)*base_resolution, int(s2)*base_resolution])
grid_pinpoints = grid_pinpoints

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
        image_tokens = ''
        instruction = ''
        source_image = None
        pos_tensors = []
        for img_path_i in range(len(input_image_path)):
            image_path = input_image_path[img_path_i]
            if image_path:
                image = Image.open(os.path.join(TOTAL_DIR,image_path)).convert('RGB')

                source_image = image.resize((1024, 1024))

                image_tensor, patch_pos_tensor = process_anyres_image(image, image_transform, grid_pinpoints, base_resolution)

                image_tensor = image_tensor.to(device, dtype=dtype)
                
                input_images.append(image_tensor)

                patch_pos_tensor = patch_pos_tensor.to(device, dtype=dtype)

                pos_tensors.append(patch_pos_tensor)

                patch_length = image_tensor.shape[0]
                for _ in range(patch_length-1):
                    image_tokens +=  BOP_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOP_TOKEN
                image_tokens += BOI_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN

            instruction += input_text_list[img_path_i]
        
        for out_img in out_image_list:
            if out_img != None:

                image_tensor, patch_pos_tensor = process_anyres_image(out_img, image_transform, grid_pinpoints, base_resolution)
                image_tensor = image_tensor.to(device, dtype=dtype)

                input_images.append(image_tensor)
                patch_pos_tensor = patch_pos_tensor.to(device, dtype=dtype)

                pos_tensors.append(patch_pos_tensor)

                patch_length = image_tensor.shape[0]
                for _ in range(patch_length-1):
                    image_tokens +=  BOP_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOP_TOKEN
                image_tokens += BOI_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN

        for out_text in out_text_list:
            instruction += out_text
        
        if len(input_images) > 1:
            image_tensor = torch.cat(input_images, dim=0)  # 在 batch 维度拼接张量
        elif len(input_images) == 1:
            image_tensor = input_images[0]  # 如果只有一张图片，直接使用该张量
        else:
            image_tensor = None

        if image_tensor != None:
            embeds_cmp_mask = torch.tensor([True]*image_tensor.shape[0]).to(device, dtype=torch.bool)
            patch_pos = [torch.cat(pos_tensors, dim=0)]
            patch_position = torch.cat(patch_pos, dim=0)
        else:
            embeds_cmp_mask = None
            patch_pos = None
            patch_position = None

        prompt = instruction_prompt.format_map({'instruction': image_tokens + instruction})

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = [tokenizer.bos_token_id] + input_ids

        input_ids = torch.tensor(input_ids).to(device, dtype=torch.long)
        ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        boi_indices = torch.where(torch.logical_or(input_ids == boi_token_id, input_ids == bop_token_id))[0].tolist()
        eoi_indices = torch.where(torch.logical_or(input_ids == eoi_token_id, input_ids == eop_token_id))[0].tolist()

        for boi_idx, eoi_idx in zip(boi_indices, eoi_indices):
            ids_cmp_mask[boi_idx + 1:eoi_idx] = True

        input_ids = input_ids.unsqueeze(0)
        ids_cmp_mask = ids_cmp_mask.unsqueeze(0)

        with torch.no_grad():
            if image_tensor != None:
                image_embeds = visual_encoder(image_tensor)
            else:
                image_embeds = None
            output = agent_model.generate(tokenizer=tokenizer,
                                        input_ids=input_ids,
                                        image_embeds=image_embeds,
                                        embeds_cmp_mask=embeds_cmp_mask,
                                        patch_positions=patch_position,
                                        ids_cmp_mask=ids_cmp_mask,
                                        num_img_gen_tokens=num_img_out_tokens)
        text = re.sub('<[^>]*>', '', output['text'])
        out_text_list.append(text)
        print(text)

        if output['has_img_output']:
            images = adapter.generate(image_embeds=output['img_gen_feat'], latent_image=source_image, num_inference_steps=50)

            save_path = os.path.join(OUTPUT_DIR, f'{real_data_list[a_index]["total_uid"]}-o-{i_step}.jpg')
            out_image_list.append(images[0])
            images[0].save(save_path)
        else:
            prompt = generation_prompt.format_map({'instruction': image_tokens + instruction + text})
            
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = [tokenizer.bos_token_id] + input_ids

            input_ids = torch.tensor(input_ids).to(device, dtype=torch.long)
            ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)

            boi_indices = torch.where(torch.logical_or(input_ids == boi_token_id, input_ids == bop_token_id))[0].tolist()
            eoi_indices = torch.where(torch.logical_or(input_ids == eoi_token_id, input_ids == eop_token_id))[0].tolist()

            for boi_idx, eoi_idx in zip(boi_indices, eoi_indices):
                ids_cmp_mask[boi_idx + 1:eoi_idx] = True

            input_ids = input_ids.unsqueeze(0)
            ids_cmp_mask = ids_cmp_mask.unsqueeze(0)

            
            with torch.no_grad():
                output = agent_model.generate(tokenizer=tokenizer,
                                        input_ids=input_ids,
                                        image_embeds=image_embeds,
                                        embeds_cmp_mask=embeds_cmp_mask,
                                        patch_positions=patch_position,
                                        ids_cmp_mask=ids_cmp_mask,
                                        num_img_gen_tokens=num_img_out_tokens)
                if output['has_img_output']:
                    images = adapter.generate(image_embeds=output['img_gen_feat'], latent_image=source_image, num_inference_steps=50)
                    save_path = os.path.join(OUTPUT_DIR, f'{real_data_list[a_index]["total_uid"]}-o-{i_step}.jpg')
                    out_image_list.append(images[0])
                    images[0].save(save_path)
                    out_image_path_list.append(save_path)
            
            i_step += 1
        save_results(real_data_list[a_index], out_text_list, out_image_path_list)
        torch.cuda.empty_cache()