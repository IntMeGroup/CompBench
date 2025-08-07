import json
import random
import os
from tqdm import tqdm

def main():
    input_jsonl_path = '/media/amax/e1efc3d3-8977-4b90-9121-3f956ab56974/huiyu/wjr/wjr/AIGI_pairs/data/pairs_difficult_mos2_multi_image_input.jsonl'
    output_train_jsonl_path = '/media/amax/e1efc3d3-8977-4b90-9121-3f956ab56974/huiyu/wjr/wjr/AIGI_pairs/data/pairs_difficult_mos2_train.jsonl'
    output_test_jsonl_path = '/media/amax/e1efc3d3-8977-4b90-9121-3f956ab56974/huiyu/wjr/wjr/AIGI_pairs/data/pairs_difficult_mos2_test.jsonl'
    swap_probability = 0.5
    train_split_ratio = 0.8

    if not os.path.exists(input_jsonl_path):
        print(f"Error: Input file not found at {input_jsonl_path}")
        return

    processed_data = []

    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as infile:
            for line_idx, line in enumerate(infile):
                try:
                    data_item = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Skipping line {line_idx + 1} due to JSON decode error: {e}. Line content: '{line.strip()}'")
                    continue

                # Corrected if condition with parentheses
                if not (isinstance(data_item, dict) and
                    'image' in data_item and
                    isinstance(data_item['image'], list) and len(data_item['image']) == 2 and
                    'conversations' in data_item and
                    isinstance(data_item['conversations'], list) and len(data_item['conversations']) >= 2 and
                    isinstance(data_item['conversations'][1], dict) and
                    data_item['conversations'][1].get('from') == 'gpt' and
                    'value' in data_item['conversations'][1]):
                    print(f"Skipping line {line_idx + 1} due to unexpected data structure: {data_item}")
                    continue
                
                original_gpt_value = data_item['conversations'][1]['value']

                # Perform swap with 50% probability
                if random.random() < swap_probability:
                    # Swap images
                    data_item['image'][0], data_item['image'][1] = data_item['image'][1], data_item['image'][0]
                    
                    if original_gpt_value == "1":
                        data_item['conversations'][1]['value'] = "2"
                    elif original_gpt_value == "2": 
                        data_item['conversations'][1]['value'] = "1"
                    else:
                        print(f"Warning: Original GPT value on line {line_idx + 1} was '{original_gpt_value}'. Setting to '2' after swap.")
                        data_item['conversations'][1]['value'] = "2"
                else:
                    # No swap, ensure GPT answer is "1"
                    data_item['conversations'][1]['value'] = "1"
                
                processed_data.append(data_item)

    except FileNotFoundError:
        print(f"Error: Input JSONL file not found at {input_jsonl_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {input_jsonl_path}: {e}")
        return

    if not processed_data:
        print("No data processed. Exiting.")
        return

    random.shuffle(processed_data)

    split_index = int(len(processed_data) * train_split_ratio)
    train_data = processed_data[:split_index]
    test_data = processed_data[split_index:]

    try:
        with open(output_train_jsonl_path, 'w', encoding='utf-8') as outfile_train:
            for item in tqdm(train_data, desc="Writing train data"):
                outfile_train.write(json.dumps(item) + '\n')
        print(f"Train data saved to {output_train_jsonl_path} ({len(train_data)} items)")
    except Exception as e:
        print(f"Error writing train data to {output_train_jsonl_path}: {e}")

    try:
        with open(output_test_jsonl_path, 'w', encoding='utf-8') as outfile_test:
            for item in tqdm(test_data, desc="Writing test data"):
                outfile_test.write(json.dumps(item) + '\n')
        print(f"Test data saved to {output_test_jsonl_path} ({len(test_data)} items)")
    except Exception as e:
        print(f"Error writing test data to {output_test_jsonl_path}: {e}")

    print("Processing complete.")

if __name__ == '__main__':
    main()
