import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu, create_attention_mask_for_mmu_vit, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
from transformers import CLIPImageProcessor
import torch.nn.functional as F
import json

from llava.llava import conversation as conversation_lib
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
SYSTEM_PROMPT_LEN = 28

TOTAL_DIR = '/mnt/workspace/zpf/OpenING'
OUTPUT_DIR = './Show-o_RAG-dev'  # Define your output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")
    
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
            print(generated_text_list)
            a_out_item = {"text": generated_text_list[index][0].strip()}
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

if __name__ == '__main__':

    config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    vision_tower_name = "openai/clip-vit-large-patch14-336"
    vision_tower =  CLIPVisionTower(vision_tower_name).to(device)
    clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

    i2t_model = Showo.from_pretrained("/mnt/workspace/zpf/show-o-w-clip-vit").to(device)
    i2t_model.eval()

    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability

    t2i_model = Showo.from_pretrained("/mnt/workspace/zpf/show-o").to(device)
    t2i_model.eval()
    mask_token_id = t2i_model.config.mask_token_id

    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = config.batch_size
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps

    print('Image token preparation done')

    saved_id = []
    for file in os.listdir(OUTPUT_DIR):
        if '.jpg' in file and file.split('-')[0] not in saved_id:
            saved_id.append(file.split('-')[0])

    data_path = os.path.join(TOTAL_DIR, "dev_data.jsonl")
    real_data_list, io_dir_list = load_data(data_path)

    print("Data Loaded!")

    for a_index, a_data in enumerate(tqdm(io_dir_list)):

        # if real_data_list[a_index]['total_uid'] in saved_id and real_data_list[a_index]['total_uid'][:4] != '0704':
        #     continue

        if real_data_list[a_index]['total_uid'] in saved_id:
            continue

        input_image_path = a_data['input_image']
        input_text_list = a_data['input_text']
        gtout_text_list = a_data['output_text']
        gtout_image_list = a_data['output_image']
        gt_out_step = len(gtout_text_list)
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
                    input_images.append(os.path.join(TOTAL_DIR,image_path))
                instruction += input_text_list[img_path_i].replace('<BEGIN>','')

            instruction = instruction + ' The answer is: ' + gtout_text_list[i_step] + '\n Please direct give answers.'
            # for out_text in out_text_list:
            #     instruction += out_text[0]

            if len(input_images) > 0:
                image_path = input_images[0]
                image_ori = Image.open(image_path).convert("RGB")
                image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
                image = image.unsqueeze(0)

                pixel_values = clip_image_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]

                image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
                images_embeddings = vision_tower(pixel_values[None])
                images_embeddings = i2t_model.mm_projector(images_embeddings)

            elif gtout_image_list[i_step]:
                image_path = gtout_image_list[i_step]
                image_ori = Image.open(os.path.join(TOTAL_DIR,image_path)).convert("RGB")
                image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
                image = image.unsqueeze(0)

                pixel_values = clip_image_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]

                image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
                images_embeddings = vision_tower(pixel_values[None])
                images_embeddings = i2t_model.mm_projector(images_embeddings)

            else:
                images_embeddings = None

            batch_size = 1

            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], instruction)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            question_input = []
            question_input.append(prompt_question.strip())

            input_ids_system = [uni_prompting.text_tokenizer(SYSTEM_PROMPT, return_tensors="pt", padding="longest").input_ids
                                    for _ in range(batch_size)]
            input_ids_system = torch.stack(input_ids_system, dim=0)
            assert input_ids_system.shape[-1] == 28
            input_ids_system = input_ids_system.to(device)
            input_ids_system = input_ids_system[0]

            input_ids = [uni_prompting.text_tokenizer(prompt, return_tensors="pt", padding="longest").input_ids
                            for prompt in question_input]

            input_ids = torch.stack(input_ids)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                    input_ids, batch_first=True, padding_value=uni_prompting.text_tokenizer.pad_token_id
            )
            input_ids = torch.tensor(input_ids).to(device).squeeze(0)
            # import pdb; pdb.set_trace()
            input_ids_llava = torch.cat([
                    (torch.ones(input_ids.shape[0], 1) *uni_prompting.sptids_dict['<|mmu|>']).to(device),
                    input_ids_system,
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                    # place your img embedding here
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                    input_ids,
            ], dim=1).long()

            text_embeddings = i2t_model.showo.model.embed_tokens(input_ids_llava)

            # Full input seq
            part1 = text_embeddings[:, :2 + SYSTEM_PROMPT_LEN, :]
            part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN:, :]
            if images_embeddings != None:
                input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)
            else:
                input_embeddings = torch.cat((part1, part2), dim=1)

            attention_mask_llava = create_attention_mask_for_mmu_vit(input_embeddings,
                                                                    system_prompt_len=SYSTEM_PROMPT_LEN)

            cont_toks_list = i2t_model.mmu_generate(input_embeddings=input_embeddings,
                                                attention_mask=attention_mask_llava[0].unsqueeze(0),
                                                max_new_tokens=100,
                                                top_k=top_k,
                                                eot_token=tokenizer.eos_token_id
                                                )
            
            cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

            text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
            out_text_list.append(text)

            prompts = [f"The prompt for this generation is: {gtout_text_list[i_step]}. The context of this task is: {instruction}."]
            image_tokens = torch.ones((len(prompts), config.model.showo.num_vq_tokens),
                                        dtype=torch.long, device=device) * mask_token_id
            input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')
            
            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                uncond_input_ids = None
            
            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
            
            with torch.no_grad():
                gen_token_ids = t2i_model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )       

            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids)

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]

            save_path = os.path.join(OUTPUT_DIR, f'{real_data_list[a_index]["total_uid"]}-o-{i_step}.jpg')
            pil_images[0].save(save_path)
                
            out_image_path_list.append(save_path)
                
            i_step += 1
            save_results(real_data_list[a_index], out_text_list, out_image_path_list)
