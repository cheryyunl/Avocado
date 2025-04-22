import torch
import os
from transformers import set_seed, logging
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images
import argparse
logging.set_verbosity_error()
import json
import sys
import re
def parse_args(args):
    parser = argparse.ArgumentParser(description="llava med")
    parser.add_argument("--output_file", type=str, help="Path to the output JSON file")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file")
    parser.add_argument("--dataset_type", type=str, help="training or validation set")
    return parser.parse_args(args)

def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def eval_model(tokenizer, model, image_processor, context_len, conv_mode, image_path, question, temperature=0.2, top_p=None, num_beams=1):
    # set_seed(0)
    # disable_torch_init()

    # model_path = os.path.expanduser(model_path)
    # model_name = model_path.split('/')[-1]
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    qs = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
    if model.config.mm_use_im_start_end:
        qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
    else:
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer,IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=1024,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    model_path = "/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/llava-med-v1.5-mistral-7b"
    model_base = None 
    conv_mode = "mistral_instruct"
    image_path = "/nfs/turbo/coe-chaijy/xuejunzh/data/validation/image/mixed-images/bbox/COCO/000000009590.png"
    
    set_seed(0)
    disable_torch_init()

    model_path = os.path.expanduser(model_path)
    model_name = model_path.split('/')[-1]
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    
    data = load_json_data(args.input_file)
    correct_predictions = 0
    total_predictions = 0
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    for item in data:
        if args.dataset_type=='validation':
            if item['data_source']=='COCO':
                categories=[
        'dining table', 'cow', 'spoon', 'potted plant', 'zebra', 'donut', 'traffic light', 'backpack', 'vase', 'sports ball', 'person', 'apple', 'chair', 'bicycle', 'pizza', 'remote', 'tennis racket', 'bench', 'boat', 'motorcycle', 'bottle', 'broccoli', 'suitcase', 'dog', 'orange', 'skis', 'cake', 'umbrella', 'cell phone', 'bowl', 'wine glass', 'surfboard', 'handbag', 'sheep', 'tie', 'banana', 'sink', 'kite', 'clock', 'cup', 'giraffe', 'bus', 'knife', 'bird', 'elephant', 'truck', 'book', 'car', 'tv', 'carrot']
            elif item['data_source']=='ADE':
                categories=[
        'plant', 'window', 'glass', 'windshield', 'vase', 'mirror', 'tree', 'ceiling', 'cabinet', 'rock', 'person', 'bag', 'chair', 'door', 'light', 'food', 'arm', 'base', 'bottle', 'brand', 'grass', 'box', 'pole', 'license plate', 'curtain', 'plate', 'mountain', 'table', 'head', 'building', 'balcony', 'shelf', 'pillow', 'column', 'shutter', 'flowerpot', 'leg', 'apron', 'sign', 'picture', 'cushion', 'flower', 'drawer', 'wheel', 'roof', 'book', 'price tag', 'car', 'rim', 'handle']
        elif args.dataset_type=='train':
            if item['data_source']=='COCO':
                categories=[
        'apple', 'backpack', 'banana', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'handbag', 'horse', 'kite', 'knife', 'motorcycle', 'orange', 'person', 'pizza', 'potted plant', 'remote', 'sheep', 'sink', 'skateboard', 'skis', 'spoon', 'sports ball', 'suitcase', 'surfboard', 'tie', 'traffic light', 'truck', 'tv', 'umbrella', 'vase', 'wine glass']
            elif item['data_source']=='ADE':
                categories=['arm', 'armchair', 'bed', 'book', 'bottle', 'box', 'building', 'cabinet', 'car', 'ceiling', 'chair', 'column', 'curtain', 'cushion', 'door', 'drawer', 'fence', 'floors', 'flower', 'glass', 'grass', 'handle', 'head', 'lamp', 'leg', 'light', 'light source', 'mirror', 'mountain', 'pane', 'person', 'picture', 'pillow', 'plant', 'plate', 'pole', 'pot', 'road', 'rock', 'seat', 'shelf', 'sign', 'sofa', 'spotlight', 'streetlight', 'table', 'tree', 'vase', 'wheel', 'window']
                
        folder_path = item['folder']
        base_path='/nfs/turbo/coe-chaijy/xuejunzh/data'
        if folder_path.startswith('/'):
            folder_path = folder_path[1:] 
        image_path = os.path.join(base_path, folder_path)  
        image = Image.open(image_path)
        index=0
        categories_str = ', '.join(categories)
        for obj in item['objects']:
            index+=1
            
            question=f"Select the single, most appropriate class for obj{index} located within the red bounding box from the following list: {categories_str}. Your response should consist solely of the class name that obj{index} belongs to, formatted as only the class name, without any extra characters or punctuation."
            response=eval_model(tokenizer, model, image_processor, context_len, conv_mode, image_path, question)
            match = re.search(r'"([^"]+)"', response)
            if match:
                prediction = match.group(1)
            else:
                prediction = response.strip().strip('.')
            prediction = prediction.rstrip('.')
            obj['prediction'] = prediction
            print("prediction:",prediction)
            if prediction.lower() == obj['name']:
                correct_predictions += 1
            total_predictions += 1
            with open(args.output_file, 'w') as f:
                json.dump(data, f, indent=4)
                
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    accuracy_output_path = os.path.join(output_dir, 'output_accuracy.json')
    
            
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")
    
    accuracy_data = {'accuracy': accuracy}
    os.makedirs(output_dir, exist_ok=True)
    with open(accuracy_output_path, 'w') as f:
        json.dump(accuracy_data, f, indent=4)
    
    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=4)

