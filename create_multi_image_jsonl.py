import csv
import json
import os
from tqdm import tqdm

def main():
    input_csv_path = '/media/amax/e1efc3d3-8977-4b90-9121-3f956ab56974/huiyu/wjr/wjr/AIGI_pairs/pair_difficult_mos2.csv'
    # Assuming prompt.txt is in the same directory as this script
    prompts_path = '/media/amax/e1efc3d3-8977-4b90-9121-3f956ab56974/huiyu/wjr/wjr/AIGI_pairs/prompt.txt' 
    output_jsonl_path = '/media/amax/e1efc3d3-8977-4b90-9121-3f956ab56974/huiyu/wjr/wjr/AIGI_pairs/data/pairs_difficult_mos2_multi_image_input.jsonl'
    image_base_path = '/mnt/data/wjr/AIGI2025/G1/' # As per example_prompt.py

    prompts = []
    try:
        with open(prompts_path, 'r') as f:
            prompts = [line.strip() for line in f.readlines()]  
        if not prompts:
            print(f"Warning: {prompts_path} is empty or could not be read.")
            # Decide if you want to exit or continue without prompts
            # return 
    except FileNotFoundError:
        print(f"Error: Prompts file not found at {prompts_path}. Please create it or specify the correct path.")
        return
    except Exception as e:
        print(f"Error reading prompts file {prompts_path}: {e}")
        return

    image_pairs_to_process = []
    try:
        with open(input_csv_path, 'r', encoding='utf-8') as infile:
            csv_reader = csv.reader(infile)
            header = next(csv_reader, None)  # Skip header row
            if header:
                print(f"Skipped header: {header}")
            else:
                print(f"Warning: Input CSV {input_csv_path} has no header or is empty.")
            
            for row_idx, row in enumerate(csv_reader):
                if len(row) >= 2:
                    image_pairs_to_process.append((row[0].strip(), row[1].strip()))
                else:
                    print(f"Skipping invalid row at index {row_idx} in {input_csv_path}: {row}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading input CSV {input_csv_path}: {e}")
        return

    if not image_pairs_to_process:
        print("No image pairs to process. Exiting.")
        return

    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for image_name1, image_name2 in tqdm(image_pairs_to_process, desc="Processing image pairs for JSONL"):
            try:
                # Extract ID from image_name1 to get the corresponding prompt
                # This logic is based on example_prompt.py: id = int(image_name1.split('.png')[0].split('/')[1])
                # Adjust if your image_name1 format is different.
                parts = image_name1.split('.png')[0].split('/')
                if len(parts) > 1:
                    image_id_str = parts[-1] # Assuming ID is the last part after splitting by '/' and before '.png'
                    image_id = int(image_id_str)
                    if 0 < image_id <= len(prompts):
                        current_prompt = prompts[image_id - 1]
                    else:
                        tqdm.write(f"Warning: Prompt ID {image_id} out of range for {image_name1}. Using default prompt.")
                        current_prompt = "Default prompt: Describe the images."
                else:
                    tqdm.write(f"Warning: Could not extract a valid ID from {image_name1} to fetch prompt. Using default prompt.")
                    current_prompt = "Default prompt: Describe the images."
            except ValueError:
                tqdm.write(f"Warning: Could not parse ID from {image_name1} as integer. Using default prompt.")
                current_prompt = "Default prompt: Describe the images."
            except Exception as e:
                tqdm.write(f"Unexpected error fetching prompt for {image_name1}: {e}. Using default prompt.")
                current_prompt = "Default prompt: Describe the images."

            question = f'Image-1: <image>\nImage-2: <image>\n Compare and jointly analyze the consistency between the text description {current_prompt} and the two images. And answer which one is better corresponding to the text description, choose from 1 and 2. Answer in one number.'
            #question = 'Image-1: <image>\nImage-2: <image>\n Compare and jointly analyze the quality, visual clarity, aesthetics, authenticity, naturalness and structural coherence of the images. And answer which one is better, choose from 1 and 2. Answer in one number.'

            # For the "gpt" answer, we'll use a placeholder. 
            # You might want to derive this from another column in your CSV if available, or leave it for the model to fill.
            # For now, let's put a generic placeholder or an example answer.
            # Example: Randomly choose or based on some logic if ImageName1 is 'better'.
            # For simplicity, I'll put a fixed placeholder answer. You can modify this.
            gpt_answer = "1" # Placeholder: e.g., assume image 1 is better or a generic response

            # Construct image paths relative to how they might be stored/accessed by the model later
            # The paths in the JSONL should be relative to the `root` specified in your dataset meta file (train_pairs.py)
            # If `image_base_path` is the root, then the paths here would just be image_name1, image_name2.
            # If your meta file's `root` is different, adjust these paths.
            # For now, assuming `image_name1` and `image_name2` are already relative paths like 'folder/image.png'
            # or just 'image.png' if they are directly under the root.
            # Given `image_base_path` was used with os.path.join in example_prompt.py, 
            # it implies `image_name1` and `image_name2` are relative paths from that base.
            # So, in the jsonl, we should store these relative paths.

            data_item = {
                "image": [image_name1, image_name2],
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": gpt_answer} # Or an empty string if model needs to generate
                ]
                # Add other fields like 'mos' if you have them for the pair
            }
            
            outfile.write(json.dumps(data_item) + '\n')

    print(f"Processing complete. JSONL output saved to {output_jsonl_path}")
    if os.path.exists('error_images.txt'): # From example_prompt.py logic
        print("Note: Some images may have failed to load during previous example_prompt.py runs, check 'error_images.txt'. This script does not load images.")

if __name__ == '__main__':
    main() 