import torch
import os
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

import argparse
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

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # process model and processor
    model_path = "/nfs/turbo/coe-chaijy/xuejunzh/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:0")

    
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
            # conversation = [
            #     {

            #     "role": "user",
            #     "content": [
            #         {"type": "text", "text": question},
            #         {"type": "image"},
            #         ],
            #     },
            # ]
            
                     
            prompt = f"[INST] <image>\n{question} [/INST]"
            #prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
            #print("inputs:",inputs)
            output = model.generate(**inputs, max_new_tokens=100)
            #print("output: ",output)
            output=processor.decode(output[0], skip_special_tokens=True)
            print(output)
            prediction = output.split("[/INST] ")[-1].strip()
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

