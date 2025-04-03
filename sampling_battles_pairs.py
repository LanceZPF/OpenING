import json
import random
import os

SAVE_FILE_NAME = './Interleaved_Arena/new_data_instance_modelAB.json'

def save_distribute_results(save_name, pk_number_per_instance):
    # 加载 baseline_models.json 文件
    with open('./Interleaved_Arena/baseline_models.json', 'r', encoding='utf-8') as f:
        baseline_models = json.load(f)
    # baseline_models = [
    #     {0: {"name": "Human"}},
    #     {1: {"name": "Gemini1.5+Flux"}},
    #     {2: {"name": "VILA-U"}},
    #     {3: {"name": "Emu3"}},
    #     {4: {"name": "anole"}},
    #     {5: {"name": "SEED-X"}},
    #     {6: {"name": "SEED-LLaMA"}},
    #     {7: {"name": "Emu2"}},
    #     {8: {"name": "Show-o"}},
    #     {9: {"name": "MiniGPT-5opening"}},
    #     {10: {"name": "MiniGPT-5mmd"}},
    #     {11: {"name": "MiniGPT-5"}},
    #     {12: {"name": "gill"}}
    # ]

    # Simplify the models dictionary creation
    models = {str(i): model for i, model in enumerate(baseline_models)}
    model_ids = list(models.keys())

    # 创建一个计算subtask和对应数据条目数的字典
    subtask2datanum = {}
    with open('./OpenING-benchmark/test_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            subtask = data['subtask_name']
            if subtask in subtask2datanum:
                subtask2datanum[subtask] += 1
            else:
                subtask2datanum[subtask] = 1

    results = []

    # 遍历 test_set.jsonl 文件
    with open('./OpenING-benchmark/test_data.jsonl', 'r', encoding='utf-8') as f:

        subtask_counter = 0
        each_subtask_data = {}

        for line in f:
            data = json.loads(line)
            data_id = data['total_uid']
            data_subtask = data['subtask_name']

            if data_subtask in each_subtask_data:
                each_subtask_data[data_subtask].append(data)
            else:
                each_subtask_data[data_subtask] = [data]

            # 处理每个subtask的数据（假设每个subtask包含40条数据）
            if len(each_subtask_data[data_subtask]) == subtask2datanum[data_subtask]:
                # 随机打乱模型列表，以确保抽样的随机性
                random.shuffle(model_ids)
                
                # 轮流分配模型对，确保所有模型在subtask内都被覆盖
                model_pairs = []
                for i in range(len(model_ids)):
                    # random.shuffle(model_ids)
                    model_A_id = model_ids[i]
                    model_B_id = model_ids[(i + 1) % len(model_ids)]  # 避免与自身配对
                    model_pairs.append((model_A_id, model_B_id))

                # 保证对subtask内的n条数据都能抽到模型 (n大于模型数量)
                random_int = random.randint(0, len(model_pairs)-1)
                for idx, data in enumerate(each_subtask_data[data_subtask]):
                    selected_pair = model_pairs[random_int % len(model_pairs)]
                    random_int += 1
                    model_A_id, model_B_id = selected_pair

                    result_entry = {
                        'data_id': data['total_uid'],
                        'model_A': {
                            'id': model_A_id,
                            'name': models[model_A_id][str(model_A_id)]['name']
                        },
                        'model_B': {
                            'id': model_B_id,
                            'name': models[model_B_id][str(model_B_id)]['name']
                        }
                    }
                    results.append(result_entry)

                # 完成当前subtask，重置数据缓存
                each_subtask_data[data_subtask] = []
                subtask_counter += 1

    print(f"一共Subtask：{subtask_counter}")

    for i in range(pk_number_per_instance-1):
        # 第二次抽样，确保结果与第一次的结果不重复
        new_results = []
        existing_pairs = {(entry['data_id'], entry['model_A']['id'], entry['model_B']['id']) for entry in results}

        for result in results:
            data_id = result['data_id']
            random.shuffle(model_ids)
            available_model_ids = model_ids[:]
            available_model_ids = set(model_ids) - {result['model_A']['id'], result['model_B']['id']}
            available_model_ids = list(available_model_ids)

            # 从可用的模型中选择新的模型对，确保与存在的结果不重复
            while True:
                model_A_id, model_B_id = random.sample(list(set(available_model_ids)), 2)              
                if (data_id, model_A_id, model_B_id) not in existing_pairs and (data_id, model_B_id, model_A_id) not in existing_pairs:
                    break

            new_result_entry = {
                'data_id': data_id,
                'model_A': {
                    'id': model_A_id,
                    'name': models[model_A_id][str(model_A_id)]['name']
                },
                'model_B': {
                    'id': model_B_id,
                    'name': models[model_B_id][str(model_B_id)]['name']
                }
            }
            new_results.append(new_result_entry)

        # 合并原有的结果和新的结果，将总数据条目数扩出为原条目的两倍
        results.extend(new_results)

    print(f"一共数据条数：{len(results)}")

    # 将结果保存到新的 JSON 文件
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"结果已成功保存到{save_name}文件中。")

def revise_model_list(model_name, new_model_name, filename):
    # 读取指定的 JSON 文件
    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 计数模型出现次数
    count = 0
    for entry in results:
        if entry['model_A']['name'] == model_name:
            entry['model_A']['name'] = new_model_name
        if entry['model_B']['name'] == model_name:
            entry['model_B']['name'] = new_model_name

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print('Done!')

def count_model_occurrences(model_name, filename):
    # 读取指定的 JSON 文件
    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 计数模型出现次数
    count = 0
    for entry in results:
        if entry['model_A']['name'] == model_name or entry['model_B']['name'] == model_name:
            count += 1
        if entry['model_A']['name'] == model_name and entry['model_B']['name'] == model_name:
            print(entry)

    return count

def count_all_model_occurrences(filename):
    # 读取指定的 JSON 文件
    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 创建一个字典来存储每个模型的出现的 total_uid
    model_counts = {entry['model_A']['name']: [] for entry in results}
    model_counts.update({entry['model_B']['name']: [] for entry in results})

    # 遍历结果，统计每个模型出现的 total_uid
    for entry in results:
        model_counts[entry['model_A']['name']].append(entry['data_id'])
        model_counts[entry['model_B']['name']].append(entry['data_id'])

    return model_counts

if __name__ == '__main__':
    save_name = SAVE_FILE_NAME
    pk_number_per_instance = 2
    save_distribute_results(save_name, pk_number_per_instance)

    # revise_model_list('seed-llama', 'SEED-LLaMA', save_name)
    # model_name_to_check = 'Human'  # 替换为你要检查的模型 ID
    # occurrences = count_model_occurrences(model_name_to_check, filename=save_name)
    # print(f"模型 {model_name_to_check} 总共出现的条目次数: {occurrences}")

    # 计算所有模型的出现次数
    all_count = 0
    model_occurrences = count_all_model_occurrences(filename=save_name)
    for model_name, count in model_occurrences.items():
        print(f"模型 {model_name} 总共出现的条目次数: {len(count)}")
        all_count += len(count)
    
    print(int(all_count/2))