import json
import os

def extract_and_filter_data(file_paths):
    results = []
    total_count = 0
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                total_count += 1
                data = json.loads(line)
                # Ensure the necessary nested data exists and is not null
                if data is None or (not (data.get('completion') and data.get('result'))):
                    continue  # Skip this entry if essential parts are missing

                completion = data['completion']
                result = data['result']
                pid = data['pid']

                if type(result) is not dict:
                    continue

                # Extract and validate required fields
                # completion_id = completion.get('id')
                context = result.get('Context')
                key_idea = result.get('Key Idea')
                method = result.get('Method')
                outcome = result.get('Outcome')
                future_impact = result.get('Future Impact')

                # Ensure all extracted fields are present and Future Impact is not 'N/A'
                if None not in [pid, context, key_idea, method, outcome, future_impact] and 'N/A' not in [context, key_idea, method, outcome, future_impact]:
                    entry = {
                        'pid': pid,
                        'context': context,
                        'key_idea': key_idea,
                        'method': method,
                        'outcome': outcome,
                        'future_impact': future_impact
                    }
                    results.append(entry)
            
    print(f"Total entries: {total_count}")
    print(f"Filtered entries: {len(results)}")

    return results

def load_meta_data(meta_file_path):
    meta_data_cache = 'meta_data_cache.json'
    
    if os.path.exists(meta_data_cache):
        with open(meta_data_cache, 'r') as cache_file:
            meta_data = json.load(cache_file)
        print(f"Loaded meta data from cache: {meta_data_cache}")
    else:
        meta_data = {}
        with open(meta_file_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                entry_id = entry.get('id')
                if entry_id:
                    meta_data[entry_id] = {
                        'venue': entry.get('venue'),
                        'year': entry.get('year'),
                        'title': entry.get('title')
                    }
        with open(meta_data_cache, 'w') as cache_file:
            json.dump(meta_data, cache_file)
        print(f"Processed and cached meta data to: {meta_data_cache}")
    
    return meta_data

# def load_meta_data(meta_file_path):
#     meta_data_cache = 'meta_data_cache.json'
    
#     # if os.path.exists(meta_data_cache):
#     #     with open(meta_data_cache, 'r') as cache_file:
#     #         meta_data = json.load(cache_file)
#     #     print(f"Loaded meta data from cache: {meta_data_cache}")
#     # else:
#     with open(meta_file_path, 'r') as file:
#         meta_data_list = json.load(file)
#     meta_data = {}
#     for entry in meta_data_list:
#         entry_id = entry.get('id')
#         if entry_id:
#             meta_data[entry_id] = {
#                 'venue': entry.get('venue'),
#                 'year': entry.get('year')
#             }
#     with open(meta_data_cache, 'w') as cache_file:
#         json.dump(meta_data, cache_file)
#     print(f"Processed and cached meta data to: {meta_data_cache}")
    
#     return meta_data

def merge_data(filtered_data, meta_data):
    for entry in filtered_data:
        entry_pid = entry['pid']
        if entry_pid in meta_data:
            entry.update(meta_data[entry_pid])
        else:
            raise ValueError(f"Meta data not found for entry: {entry_pid}")
    return filtered_data

def save_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')
    print(f"Data saved to {filename}")

def count_total_lines(file_paths):
    total_lines = 0
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                total_lines += 1
    return total_lines

if __name__ == "__main__":

    file_paths = [
                  "../../raw_data/2024-05-30_11-54-04.jsonl",
                  "../../raw_data/2024-05-30_05-00-00.jsonl",
                  "../../raw_data/2024-05-29_21-45-15.jsonl",
                  "../../raw_data/2024-05-29_17-58-51.jsonl",
                  "../../raw_data/2024-05-29_14-20-46.jsonl",
                  "../../raw_data/2024-05-29_13-45-50.jsonl"]

    meta_file_path = "../../raw_data/oag_publication_selected_0525.json"

    print(count_total_lines(file_paths))
    filtered_data = extract_and_filter_data(file_paths)
    meta_data = load_meta_data(meta_file_path)
    merged_data = merge_data(filtered_data, meta_data)

    save_to_jsonl(merged_data, 'filtered_data_with_meta_info.jsonl')

    print(len(merged_data))
    print(merged_data[0])
    print()
    print(merged_data[1])
    print()
