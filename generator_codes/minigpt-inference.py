import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torchvision import transforms
from model import MiniGPT5_Model
from train_eval import ModelArguments, DataArguments, TrainingArguments
from PIL import Image
import transformers
import torch
import matplotlib.pyplot as plt
import textwrap
from lightning.pytorch import seed_everything


from tqdm import tqdm
import json

TOTAL_DIR = '/mnt/workspace/zpf/OpenING'
OUTPUT_DIR = './MiniGPT-5mmd_dumb-dev'  # Define your output directory
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
        a_out_item = {"text": generated_text_list[index].replace("[IMG0]", "")}
        if image_out_list[index] is not None:
            a_out_item["image"] = f'{data_uid}-o-{index}.jpg'
        else:
            a_out_item["image"] = None
        saved_json['conversations'][1]['output'].append(a_out_item)

        # Save generated image
        if image_out_list[index] is not None:
            image_path = os.path.join(OUTPUT_DIR, f'{data_uid}-o-{index}.jpg')
            image_out_list[index].save(image_path)

    with open(jsonl_path, mode='w', encoding='utf-8') as writer:
        json.dump(saved_json, writer, indent=4)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

if __name__ == "__main__":
    seed_everything(42)

    saved_id = []
    for file in os.listdir(OUTPUT_DIR):
        if '.jpg' in file and file.split('-')[0] not in saved_id:
            saved_id.append(file.split('-')[0])

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if isinstance(training_args.gpus, str):
        training_args.gpus = [int(x) for x in training_args.gpus.split(',')]

    stage1_ckpt = model_args.stage1_weight
    stage2_ckpt = training_args.test_weight

    minigpt5 = MiniGPT5_Model.load_from_checkpoint(stage1_ckpt, strict=False, map_location="cpu", encoder_model_config=model_args, **vars(training_args))
    finetuned_state_dict = torch.load(stage2_ckpt, map_location="cpu")['state_dict']
    minigpt5.load_state_dict(finetuned_state_dict, strict=False)
    minigpt5.to(torch.device("cuda:0"), torch.float16)
    minigpt5.eval()

    input_vis_processor = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    input_images = None

    data_path = os.path.join(TOTAL_DIR, 'dev_data.jsonl')
    real_data_list, io_dir_list = load_data(data_path)

    for a_index, a_data in enumerate(io_dir_list):
        if real_data_list[a_index]['total_uid'] in saved_id:
            continue

        input_image_path = a_data['input_image']
        gt_out_step = len(a_data['output_text'])
        out_image_list = []
        out_text_list = []

        for i in range(gt_out_step):
            if input_image_path:
                input_images = []
                for img_path in input_image_path:
                    if img_path:
                        try:
                            input_image = Image.open(os.path.join(TOTAL_DIR, img_path)).convert("RGB")
                            input_image = expand2square(input_image, (255, 255, 255))
                            input_image = input_vis_processor(input_image)
                            input_image = input_image.unsqueeze(0).to("cuda:0")
                            input_images.append(input_image)
                        except Exception as e:
                            print(img_path)
                            input_images = None

                # '''
                for out_img in out_image_list:
                    try:
                        out_img = expand2square(out_img, (255, 255, 255))
                        out_img = input_vis_processor(out_img)
                        out_img = out_img.unsqueeze(0).to("cuda:0")
                        input_images.append(out_img)
                    except Exception as e:
                        print(out_img)

                if len(input_images) > 0:
                    input_images = torch.cat(input_images, dim=0)
                    input_images = input_images.to(torch.device("cuda:0"))
                else:
                    input_images = None
                # '''

            system_prompt="Give the following information in text and <Img>ImageContent</Img> format. You will be able to see the images once I provide it to you. Please understanding input and generate images and text."
            utterance = ""

            for i_text in a_data['input_text']:
                utterance += i_text.replace('<image>','<Img><ImageHere></Img>') + "\n"
            
            # '''
            for o_text in out_text_list:
                utterance += o_text.replace('[IMG0]', '<Img><ImageHere></Img>') + "\n"
            # '''

            utterance = system_prompt + f"###Human:{utterance} Tell me the next step with image. ###Assistant:"

            with torch.inference_mode():
                with torch.autocast("cuda:0"):
                    text_out, image_out = minigpt5.generate(utterance, input_images)
            generated_text = text_out.replace("###", "")
            # wrapped_generated_text = textwrap.fill(generated_text, 50)
            out_text_list.append(generated_text)
            out_image_list.append(image_out)

        save_results(real_data_list[a_index], out_text_list, out_image_list)  # Save results