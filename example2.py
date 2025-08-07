import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import csv
import os
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    try:
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    except FileNotFoundError:
        # Reduced verbosity for progress bar
        # print(f"Error: Image file not found at {image_file}")
        with open('error_images.txt', 'a') as f:
            f.write(f"{image_file}\n")
        return None
    except Exception as e:
        # Reduced verbosity for progress bar
        # print(f"Error loading image {image_file}: {e}")
        with open('error_images.txt', 'a') as f:
            f.write(f"{image_file}\n")
        return None

# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
# path = '/home/wangjiarui/AIGI_2025/OpenGVLab/InternVL2_5-8B'
path = '/home/wangjiarui/InternVL/internvl_chat/InternVL3-9B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=False)

# --- Main script logic modified --- 
input_csv_path = '/home/wangjiarui/AIGI_pairs/pairs_mos1.csv'
image_base_path = '/home/wangjiarui/AIGI2025/'
output_csv_path = 'AIGI_pairs_mos1_internvl3.csv'

question_template = 'Image-1: <image>\nImage-2: <image>\n Compare and jointly analyze the quality, visual clarity, aesthetics, authenticity, naturalness and structural coherence of the images. And answer which one is better, choose from 1 and 2. Answer in one number.'

# Read all rows first to get the total for tqdm
image_pairs_to_process = []
try:
    with open(input_csv_path, 'r', encoding='utf-8') as infile:
        csv_reader = csv.reader(infile)
        header = next(csv_reader, None) # Skip header row
        if header:
            print(f"Skipped header: {header}")
        else:
            print("Warning: Input CSV has no header or is empty.")
        for row in csv_reader:
            if len(row) >= 2:
                image_pairs_to_process.append((row[0].strip(), row[1].strip()))
            else:
                print(f"Skipping invalid row: {row}")
except FileNotFoundError:
    print(f"Error: Input CSV file not found at {input_csv_path}")
    image_pairs_to_process = [] # Ensure it's empty if file not found
except Exception as e:
    print(f"An unexpected error occurred while reading input CSV: {e}")
    image_pairs_to_process = []

if not image_pairs_to_process:
    print("No image pairs to process. Exiting.")
else:
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['ImageName1', 'ImageName2', 'Answer'])

        # Wrap the iteration with tqdm for a progress bar
        for image_name1, image_name2 in tqdm(image_pairs_to_process, desc="Processing image pairs"):
            image_path1 = os.path.join(image_base_path, image_name1)
            image_path2 = os.path.join(image_base_path, image_name2)

            # Reduced verbosity for progress bar
            # print(f"Processing pair: {image_name1}, {image_name2}")

            pixel_values1 = load_image(image_path1, max_num=12)
            pixel_values2 = load_image(image_path2, max_num=12)

            if pixel_values1 is None or pixel_values2 is None:
                # Error already printed by load_image
                tqdm.write(f"Skipping pair due to image loading error: {image_name1}, {image_name2}")
                csv_writer.writerow([image_name1, image_name2, 'Error loading images'])
                continue
            
            pixel_values1 = pixel_values1.to(torch.bfloat16).cuda()
            pixel_values2 = pixel_values2.to(torch.bfloat16).cuda()
            
            pixel_values_combined = torch.cat((pixel_values1, pixel_values2), dim=0)
            num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

            try:
                chat_output = model.chat(tokenizer, pixel_values_combined, question_template, generation_config,
                                       num_patches_list=num_patches_list,
                                       history=None, return_history=False)
                
                if isinstance(chat_output, tuple) or isinstance(chat_output, list):
                    if len(chat_output) > 0:
                        response = chat_output[0]
                        # if len(chat_output) > 1:
                        #     tqdm.write(f"  Debug: Additional chat output elements: {chat_output[1:]}") # Use tqdm.write
                    else:
                        tqdm.write(f"  Warning: model.chat returned an empty tuple/list for {image_name1}, {image_name2}. Output: {chat_output}")
                        response = "Error: Empty chat output"
                elif isinstance(chat_output, str):
                    response = chat_output
                else:
                    tqdm.write(f"  Warning: model.chat returned unexpected type for {image_name1}, {image_name2}. Type: {type(chat_output)}, Output: {chat_output}")
                    response = str(chat_output)

                # tqdm.write(f'  Model Response: {response}') # Optional: if you want to see every response with progress bar
                csv_writer.writerow([image_name1, image_name2, response])
            except Exception as e:
                tqdm.write(f"Error during model.chat for {image_name1}, {image_name2}: {e}")
                csv_writer.writerow([image_name1, image_name2, f'Error during model inference: {e}'])

    print(f"Processing complete. Results saved to {output_csv_path}")