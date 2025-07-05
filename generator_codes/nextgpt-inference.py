import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from model.anyToImageVideoAudio import NextGPTModel
import torch
import json
from config import *
import matplotlib.pyplot as plt
from diffusers.utils import export_to_video
import scipy
import time

from tqdm import tqdm

TOTAL_DIR = '/mnt/workspace/zpf/OpenING'
OUTPUT_DIR = './NExT-GPT_output'  # Define your output directory
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

    for index in range(len(generated_text_list)):
        a_out_item = {"text": generated_text_list[index].replace("[IMG1][IMG2][IMG3][IMG4][IMG5][IMG6][IMG7]", "")}
        if index < len(image_out_list):
            if image_out_list[index] is not None:
                a_out_item["image"] = f'{data_uid}-o-{index}.jpg'
                # image_path = os.path.join(OUTPUT_DIR, f'{data_uid}-o-{index}.jpg')
                # image_out_list[index].save(image_path)
            else:
                a_out_item["image"] = None
        else:
            a_out_item["image"] = None

        saved_json['conversations'][1]['output'].append(a_out_item)

    with open(jsonl_path, mode='w', encoding='utf-8') as writer:
        json.dump(saved_json, writer, indent=4)

def predict(
        input,
        image_path=None,
        audio_path=None,
        video_path=None,
        thermal_path=None,
        max_tgt_len=200,
        top_p=10.0,
        temperature=0.1,
        history=None,
        modality_cache=None,
        filter_value=-float('Inf'), min_word_tokens=0,
        gen_scale_factor=10.0, max_num_imgs=1,
        stops_id=None,
        load_sd=True,
        generator=None,
        guidance_scale_for_img=7.5,
        num_inference_steps_for_img=50,
        guidance_scale_for_vid=7.5,
        num_inference_steps_for_vid=50,
        max_num_vids=1,
        height=320,
        width=576,
        num_frames=24,
        guidance_scale_for_aud=7.5,
        num_inference_steps_for_aud=50,
        max_num_auds=1,
        audio_length_in_s=9,
        ENCOUNTERS=1,
):
    if image_path is None and audio_path is None and video_path is None and thermal_path is None:
        # return [(input, "There is no input data provided! Please upload your data and start the conversation.")]
        print('no image, audio, video, and thermal are input')
    else:
        print(
            f'[!] image path: {image_path}\n[!] audio path: {audio_path}\n[!] video path: {video_path}\n[!] thermal path: {thermal_path}')

    # prepare the prompt
    prompt_text = ''
    if history != None:
        for idx, (q, a) in enumerate(history):
            if idx == 0:
                prompt_text += f'{q}\n### Assistant: {a}\n###'
            else:
                prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
        prompt_text += f'### Human: {input}'
    else:
        prompt_text += f'### Human: {input}'

    # print('prompt_text: ', prompt_text)

    response = model.generate({
        'prompt': prompt_text,
        'image_paths': image_path if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_tgt_len,
        'modality_embeds': modality_cache,
        'filter_value': filter_value, 'min_word_tokens': min_word_tokens,
        'gen_scale_factor': gen_scale_factor, 'max_num_imgs': max_num_imgs,
        'stops_id': stops_id,
        'load_sd': load_sd,
        'generator': generator,
        'guidance_scale_for_img': guidance_scale_for_img,
        'num_inference_steps_for_img': num_inference_steps_for_img,

        'guidance_scale_for_vid': guidance_scale_for_vid,
        'num_inference_steps_for_vid': num_inference_steps_for_vid,
        'max_num_vids': max_num_vids,
        'height': height,
        'width': width,
        'num_frames': num_frames,

        'guidance_scale_for_aud': guidance_scale_for_aud,
        'num_inference_steps_for_aud': num_inference_steps_for_aud,
        'max_num_auds': max_num_auds,
        'audio_length_in_s': audio_length_in_s,
        'ENCOUNTERS': ENCOUNTERS,

    })
    return response


if __name__ == '__main__':

    saved_id = []
    for file in os.listdir(OUTPUT_DIR):
        if '.jpg' in file and file.split('-')[0] not in saved_id:
            saved_id.append(file.split('-')[0])

    # init the model
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)
    args = {'model': 'nextgpt',
            'nextgpt_ckpt_path': '/mnt/workspace/zpf/NExT-GPT/ckpt/delta_ckpt/nextgpt/7b_tiva_v0/',
            'max_length': 128,
            'stage': 3,
            'root_dir': '/mnt/workspace/zpf/NExT-GPT/',
            'mode': 'validate',
            }
    args.update(load_config(args))

    model = NextGPTModel(**args)
    delta_ckpt = torch.load(os.path.join(args['nextgpt_ckpt_path'], 'pytorch_model.pt'), map_location=torch.device('cuda'))
    # print(delta_ckpt)
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.eval().half().cuda()
    # model = model.eval().cuda()
    print(f'[!] init the 7b model over ...')

    """Override Chatbot.postprocess"""
    max_tgt_length = 150
    top_p = 1.0
    temperature = 0.4
    modality_cache = None

    data_path = os.path.join(TOTAL_DIR, 'test_data.jsonl')
    real_data_list, io_dir_list = load_data(data_path)

    for a_index, a_data in enumerate(io_dir_list):
        if real_data_list[a_index]['total_uid'] in saved_id:
            continue
        input_image_path = a_data['input_image']
        gt_out_step = len(a_data['output_text'])
        out_image_list = []
        out_text_list = []

        i_step = 0
        while i_step < gt_out_step:
            if input_image_path:
                input_images = []
                for img_path in input_image_path:
                    if img_path:
                        input_images.append(os.path.join(TOTAL_DIR, img_path))

                # '''
                for out_img in out_image_list:
                    input_images.append(out_img)

            utterance = "Show me an image: "

            for i_text in a_data['input_text']:
                utterance += i_text.strip() + "\n"
            
            # '''
            for o_text in out_text_list:
                utterance += o_text.strip() + "\n"
            # '''

            history = []

            try:
                output = predict(input=utterance, image_path=input_images, history=history,
                        max_tgt_len=max_tgt_length, top_p=top_p,
                        temperature=temperature, modality_cache=modality_cache,
                        filter_value=-float('Inf'), min_word_tokens=10,
                        gen_scale_factor=4.0, max_num_imgs=1,
                        stops_id=[[835]],
                        load_sd=True,
                        generator=g_cuda,
                        guidance_scale_for_img=7.5,
                        num_inference_steps_for_img=50,
                        guidance_scale_for_vid=7.5,
                        num_inference_steps_for_vid=50,
                        max_num_vids=1,
                        height=320,
                        width=576,
                        num_frames=24,
                        ENCOUNTERS=1
                        )
            except Exception as e:
                print(f"ERROR: {input_images}")
                continue

            for i_o in output:
                if isinstance(i_o, str):
                    out_text_list.append(i_o)
                elif 'img' in i_o.keys():
                    for m in i_o['img']:
                        if isinstance(m, str):
                            out_text_list.append(m)
                        else:
                            save_image_path = os.path.join(OUTPUT_DIR, f'{real_data_list[a_index]["total_uid"]}-o-{i_step}.jpg')
                            if save_image_path in out_image_list:
                                i_step+=1
                                save_image_path = os.path.join(OUTPUT_DIR, f'{real_data_list[a_index]["total_uid"]}-o-{i_step}.jpg')
                            out_image_list.append(save_image_path)
                            m[0].save(save_image_path)
                else:
                    pass
            
            i_step += 1

        save_results(real_data_list[a_index], out_text_list, out_image_list)  # Save results

    # print("output: ", output)

