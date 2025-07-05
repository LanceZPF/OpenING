import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import torch
import argparse
from PIL import Image
from pathlib import Path
from chameleon.inference.chameleon import ChameleonInferenceModel, Options
from constants import (
    MODEL_7B_PATH,
    TOKENIZER_TEXT_PATH,
    TOKENIZER_IMAGE_CFG_PATH,
    TOKENIZER_IMAGE_PATH,
)
from typing import List, Dict, Tuple

from tqdm import tqdm

TOTAL_DIR = '/mnt/workspace/zpf/OpenING'
OUTPUT_DIR = './anole_output'  # Define your output directory
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

def split_token_sequence(
    tokens: torch.LongTensor,
    boi: int,
    eoi: int
) -> List[Tuple[str, torch.LongTensor]]:
    """
    Split a sequence of tokens into text and image segments.
    
    Args:
        tokens (torch.LongTensor): The token sequence.
        boi (int): Begin of image token.
        eoi (int): End of image token.
    
    Returns:
        List[Tuple[str, torch.LongTensor]]: List of tuples indicating segment type and tokens.
    """
    batch_size, _ = tokens.shape
    assert batch_size == 1, "Batch size must be 1"
    
    device = tokens.device
    tokens = tokens[0]  # remove batch dimension
    tokens = tokens.to(device)
    segments = []
    current_segment = []
    in_image_seg = False

    for token in tokens:
        if token == boi:
            # if entering an image segment, save the current text segment (if any)
            if current_segment:
                segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
                current_segment = []
            in_image_seg = True
        elif token == eoi and in_image_seg:
            # if exiting an image segment, save the current image segment
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
            current_segment = []
            in_image_seg = False
        else:
            current_segment.append(token)
    # save any remaining tokens
    if current_segment:
        if in_image_seg:
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
        else:
            segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
    return segments

def main(args: argparse.Namespace):

    saved_id = []
    for file in os.listdir(OUTPUT_DIR):
        if '.jpg' in file and file.split('-')[0] not in saved_id:
            saved_id.append(file.split('-')[0])

    """Main function to generate and process model output."""
    # Load Chameleon model
    model = ChameleonInferenceModel(
        MODEL_7B_PATH.as_posix(),
        TOKENIZER_TEXT_PATH.as_posix(),
        TOKENIZER_IMAGE_CFG_PATH.as_posix(),
        TOKENIZER_IMAGE_PATH.as_posix(),
    )
    # Print model configuration
    print(f"Model path: {MODEL_7B_PATH}")
    print(f"Text tokenizer path: {TOKENIZER_TEXT_PATH}")
    print(f"Image tokenizer config path: {TOKENIZER_IMAGE_CFG_PATH}")
    print(f"Image tokenizer path: {TOKENIZER_IMAGE_PATH}")

    # Generate options
    options = Options()
    # Prepare prompt
    input_path: Path = Path(args.input)

    data_path = os.path.join(TOTAL_DIR, input_path)
    real_data_list, io_dir_list = load_data(data_path)

    for a_index, a_data in enumerate(tqdm(io_dir_list)):
        if real_data_list[a_index]['total_uid'] in saved_id:
            continue
        input_image_path = a_data['input_image']
        input_text_list = a_data['input_text']
        gt_out_step = len(a_data['output_text'])
        out_image_list = []
        out_text_list = []

        batch_prompt_ui = [[]]
        for input_seg in range(len(input_text_list)):
            batch_prompt_ui[0] += [
                {"type": "text", "value": input_text_list[input_seg].strip()}
            ]
            img_path = input_image_path[input_seg]
            if img_path:
                abs_path: Path = os.path.abspath(os.path.join(TOTAL_DIR, img_path))
                batch_prompt_ui[0] += [
                    {"type": "image", "value": f"file:{abs_path}"},
                ]
        # generate
        try:
            tokens: torch.LongTensor = model.generate(
                batch_prompt_ui=batch_prompt_ui,
                options=options
            )
        except Exception as e:
            print(f"ERROR: {input_images}")
            continue
        
        # split
        boi, eoi = model.vocab.begin_image, model.vocab.end_image   # 8197(boi), 8196(eoi)
        segments = split_token_sequence(tokens, boi, eoi)
        
        # decode
        os.makedirs(args.save_dir, exist_ok=True)
        img_id = 0
        for seg_id, (seg_type, seg_tokens) in enumerate(segments):
            if seg_type == "image_seg":
                assert seg_tokens.shape[1] == 1024
                img: Image = model.decode_image(seg_tokens)[0]
                
                image_path = os.path.join(OUTPUT_DIR, f'{real_data_list[a_index]["total_uid"]}-o-{max(len(out_text_list) - 1, 0)}.jpg')
                if image_path not in out_image_list:
                    img_id = max(len(out_text_list) - 1, 0)
                else:
                    img_id += 1

                image_path = os.path.join(OUTPUT_DIR, f'{real_data_list[a_index]["total_uid"]}-o-{img_id}.jpg')
                img.save(image_path)
                out_image_list.append(image_path)
                # print(f"<img: {image_path}>")
            else:
                assert seg_type == "text_seg"
                decoded_text = model.decode_text(seg_tokens)[0]
                out_text_list.append(decoded_text)
                # print(decoded_text)
                
        save_results(real_data_list[a_index], out_text_list, out_image_list)  # Save results

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate interleaved image-text content based on text instructions.")
    parser.add_argument("-i", "--input", type=str, default="test_data.jsonl", help="The multimodal input file.")
    parser.add_argument("-s", "--save_dir", type=str, default=OUTPUT_DIR, help="The directory to save the generated images.")
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)
