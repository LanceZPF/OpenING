import json
import csv
from collections import defaultdict

def tie2ABdiv(results):
    # 初始化计数器
    model_wins = defaultdict(lambda: defaultdict(int))
    model_matches = defaultdict(lambda: defaultdict(int))

    # 遍历比赛结果
    for result in results:
        model_A = result['model_A']['name']
        model_B = result['model_B']['name']
        winner = result['winner']

        model_matches[model_A][model_B] += 1
        model_matches[model_B][model_A] += 1

        if winner == 'A':
            model_wins[model_A][model_B] += 1
        elif winner == 'B':
            model_wins[model_B][model_A] += 1
        elif winner == 'Tie(A)':
            model_wins[model_A][model_B] += 1
        elif winner == 'Tie(B)':
            model_wins[model_B][model_A] += 1

    # 计算总体胜场数和比赛场次
    total_wins = {}
    total_matches = {}
    for model in model_matches:
        total_wins[model] = sum(model_wins[model].values())
        total_matches[model] = sum(model_matches[model].values())

    # 计算总体胜率
    overall_win_rates = {model: total_wins[model] / total_matches[model] if total_matches[model] > 0 else 0 for model in model_matches}

    # 按总体胜率排序模型
    sorted_models = sorted(overall_win_rates, key=overall_win_rates.get, reverse=True)

    # 输出胜率
    print("Model Win Rates:")
    for model in sorted_models:
        print(f"{model}: {overall_win_rates[model]:.2%}")

    # 写入 CSV 文件
    with open('./Interleaved_Arena/model_win_rates.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Model'] + sorted_models
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for model_A in sorted_models:
            row = {'Model': model_A}
            for model_B in sorted_models:
                if model_A != model_B:
                    matches = model_matches[model_A][model_B]
                    if matches > 0:
                        wins = model_wins[model_A][model_B]
                        win_rate = wins / matches
                        row[model_B] = f"{win_rate:.2%}"
                    else:
                        row[model_B] = 'N/A'
                else:
                    row[model_B] = 'N/A'
            writer.writerow(row)

def without_tie(results):
    # 初始化计数器
    model_wins = defaultdict(int)
    model_matches = defaultdict(int)

    # 遍历比赛结果
    for result in results:
        model_A = result['model_A']['name']
        model_B = result['model_B']['name']
        winner = result['winner']

        # 更新胜场数
        if winner == 'A':
            model_wins[model_A] += 1
            model_matches[model_A] += 1
            model_matches[model_B] += 1
        elif winner == 'B':
            model_wins[model_B] += 1
            model_matches[model_A] += 1
            model_matches[model_B] += 1

    # 计算总体胜率
    overall_win_rates = {model: model_wins[model] / model_matches[model] if model_matches[model] > 0 else 0 for model in model_matches}

    # 按总体胜率排序模型
    sorted_models = sorted(overall_win_rates, key=overall_win_rates.get, reverse=True)

    # 输出胜率
    print("Model Win Rates (Without Tie):")
    for model in sorted_models:
        print(f"{model}: {overall_win_rates[model]:.2%}")

    # 写入 CSV 文件
    with open('./Interleaved_Arena/model_win_rates_notie.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Model'] + sorted_models
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for model_A in sorted_models:
            row = {'Model': model_A}
            for model_B in sorted_models:
                if model_A != model_B:
                    # 计算与特定对手的胜率
                    matches = 0
                    wins = 0
                    for result in results:
                        if (result['model_A']['name'] == model_A and result['model_B']['name'] == model_B) or \
                           (result['model_A']['name'] == model_B and result['model_B']['name'] == model_A):
                            winner = result['winner']
                            if winner == 'A' and result['model_A']['name'] == model_A:
                                wins += 1
                            elif winner == 'B' and result['model_B']['name'] == model_A:
                                wins += 1
                            elif winner == 'A' and result['model_B']['name'] == model_A:
                                wins += 0  # 对方胜利
                            elif winner == 'B' and result['model_A']['name'] == model_A:
                                wins += 0  # 对方胜利
                            matches += 1
                    if matches > 0:
                        win_rate = wins / matches
                        row[model_B] = f"{win_rate:.2%}"
                    else:
                        row[model_B] = 'N/A'
                else:
                    row[model_B] = 'N/A'
            writer.writerow(row)

def with_tie(results):
    # 初始化计数器
    model_wins = defaultdict(int)
    model_ties = defaultdict(int)
    model_matches = defaultdict(int)

    # 遍历比赛结果
    for result in results:
        model_A = result['model_A']['name']
        model_B = result['model_B']['name']
        winner = result['winner']

        # 更新比赛场次
        model_matches[model_A] += 1
        model_matches[model_B] += 1

        # 更新胜场数和平局数
        if winner == 'A':
            model_wins[model_A] += 1
        elif winner == 'B':
            model_wins[model_B] += 1
        elif 'Tie' in winner:
            model_ties[model_A] += 1
            model_ties[model_B] += 1

    # 计算总体胜率，平局算作半个胜利
    overall_win_rates = {}
    for model in model_matches:
        total_matches = model_matches[model]
        total_wins = model_wins[model]
        total_ties = model_ties[model]
        overall_win_rates[model] = total_wins / total_matches if total_matches > 0 else 0

    # 按总体胜率排序模型
    sorted_models = sorted(overall_win_rates, key=overall_win_rates.get, reverse=True)

    # 输出胜率
    print("Model Win Rates (With Tie):")
    for model in sorted_models:
        print(f"{model}: {overall_win_rates[model]:.2%}")

    # 写入 CSV 文件
    with open('./Interleaved_Arena/model_win_rates_withtie0.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Model'] + sorted_models
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for model_A in sorted_models:
            row = {'Model': model_A}
            for model_B in sorted_models:
                if model_A != model_B:
                    # 计算与特定对手的胜率
                    matches = 0
                    wins = 0
                    ties = 0
                    for result in results:
                        if ((result['model_A']['name'] == model_A and result['model_B']['name'] == model_B) or
                            (result['model_A']['name'] == model_B and result['model_B']['name'] == model_A)):
                            winner = result['winner']
                            if winner == 'A' and result['model_A']['name'] == model_A:
                                wins += 1
                            elif winner == 'B' and result['model_B']['name'] == model_A:
                                wins += 1
                            elif 'Tie' in winner:
                                ties += 1
                            matches += 1
                    if matches > 0:
                        win_rate = wins / matches
                        row[model_B] = f"{win_rate:.2%}"
                    else:
                        row[model_B] = 'N/A'
                else:
                    row[model_B] = 'N/A'
            writer.writerow(row)

def with_tie_equaldiv(results):
    # 初始化计数器
    model_wins = defaultdict(int)
    model_ties = defaultdict(int)
    model_matches = defaultdict(int)

    # 遍历比赛结果
    for result in results:
        model_A = result['model_A']['name']
        model_B = result['model_B']['name']
        winner = result['winner']

        # 更新比赛场次
        model_matches[model_A] += 1
        model_matches[model_B] += 1

        # 更新胜场数和平局数
        if winner == 'A':
            model_wins[model_A] += 1
        elif winner == 'B':
            model_wins[model_B] += 1
        elif 'Tie' in winner:
            model_ties[model_A] += 1
            model_ties[model_B] += 1

    # 计算总体胜率，平局算作半个胜利
    overall_win_rates = {}
    for model in model_matches:
        total_matches = model_matches[model]
        total_wins = model_wins[model]
        total_ties = model_ties[model]
        overall_win_rates[model] = (total_wins + 0.5*total_ties) / total_matches if total_matches > 0 else 0

    # 按总体胜率排序模型
    sorted_models = sorted(overall_win_rates, key=overall_win_rates.get, reverse=True)

    # 输出胜率
    print("Model Win Rates (With Tie):")
    for model in sorted_models:
        print(f"{model}: {overall_win_rates[model]:.2%}")

    # 写入 CSV 文件
    with open('./Interleaved_Arena/model_win_rates_withtie05.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Model'] + sorted_models
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for model_A in sorted_models:
            row = {'Model': model_A}
            for model_B in sorted_models:
                if model_A != model_B:
                    # 计算与特定对手的胜率
                    matches = 0
                    wins = 0
                    ties = 0
                    for result in results:
                        if ((result['model_A']['name'] == model_A and result['model_B']['name'] == model_B) or
                            (result['model_A']['name'] == model_B and result['model_B']['name'] == model_A)):
                            winner = result['winner']
                            if winner == 'A' and result['model_A']['name'] == model_A:
                                wins += 1
                            elif winner == 'B' and result['model_B']['name'] == model_A:
                                wins += 1
                            elif 'Tie' in winner:
                                ties += 1
                            matches += 1
                    if matches > 0:
                        win_rate = (wins + 0.5 * ties) / matches
                        row[model_B] = f"{win_rate:.2%}"
                    else:
                        row[model_B] = 'N/A'
                else:
                    row[model_B] = 'N/A'
            writer.writerow(row)

def load_evaluation_results(file_path):
    """
    Loads evaluation results from a JSON file and compiles them into a dictionary.
    The keys are formatted as f"{data_id}_{model_A_id}_{model_B_id}".
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        results = json.load(file)
    
    compiled_results = {}
    for item in results:
        data_id = item['data_id']
        model_A_id = item['model_A']['id']
        model_B_id = item['model_B']['id']
        winner = item

        # Construct the key
        key = f"{data_id}_{model_A_id}_{model_B_id}"
        compiled_results[key] = winner

    return compiled_results

def decode_complied_results(compiled_results):
    """
    Decodes the compiled results into a list of dictionaries.
    Each dictionary contains 'data_id', 'model_A_id', 'model_B_id', and 'winner'.
    """
    decoded_results = []
    for key, winner in compiled_results.items():
        decoded_results.append(compiled_results[key])
    return decoded_results, compiled_results

if __name__ == '__main__':
    # Load the match results from the JSON file
    sampled_pk_file_name = './Interleaved_Arena/data_instance_modelAB.json'
    result_file_name = './Interleaved_Arena/intjudge_rag-judge_modelAB_results.json'
    # sampled_pk_file_name = './Interleaved_Arena/data_instance_modelAB_new.json'
    # result_file_name = './Interleaved_Arena/intjudge_rag-judge_modelAB_results_new.json'
    # with open('./judge_modelAB_results.json', 'r') as file:
    with open(sampled_pk_file_name, 'r') as file:
        pk_meta = json.load(file)

    results, compiled_results = decode_complied_results(load_evaluation_results(result_file_name))

    for pk_pair in pk_meta:
        data_id = pk_pair['data_id']
        model_A_id = pk_pair['model_A']['id']
        model_B_id = pk_pair['model_B']['id']

    print('Force Dividing Tie Into A or B (FDT)')
    tie2ABdiv(results)
    print('\n\n')

    print('Without Tie (Only count the certain A and B)')
    without_tie(results)
    print('\n\n')

    print('With Tie (Count tie as 0 point)')
    with_tie(results)
    print('\n\n')

    print('With Tie (Count tie as 0.5 point)')
    with_tie_equaldiv(results)
    print('\n\n')

