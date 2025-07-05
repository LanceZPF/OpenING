import json

# UNSEEN_MODELS = ['anole', 'GPT-4+DALL-E3', 'SEED-LLaMA', 'NExT-GPT', 'VILA-U', 'Emu3', 'MiniGPT-5mmd']
UNSEEN_MODELS = ['anole', 'GPT-4+DALL-E3', 'SEED-LLaMA', 'NExT-GPT']

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
        model_A_id = item['model_A']['name']
        model_B_id = item['model_B']['name']
        winner = item['winner']

        # Construct the key
        key = f"{data_id}_{model_A_id}_{model_B_id}"
        compiled_results[key] = winner

    return compiled_results

def simplify_winner(winner):
    """
    Simplifies the winner string:
    - Converts 'Tie(A)' and 'Tie(B)' to 'Tie'
    - Leaves 'A' and 'B' unchanged
    """
    if 'tie' in winner.lower():
        return 'Tie'
    else:
        return winner

def force_divide_tie(winner):
    """
    Forces tie results into either 'A' or 'B':
    - 'Tie(A)' becomes 'A'
    - 'Tie(B)' becomes 'B'
    - 'Tie' (if present) can be assigned randomly or based on a rule
    """
    if winner == 'Tie(A)':
        return 'A'
    elif winner == 'Tie(B)':
        return 'B'
    elif winner == 'Tie':
        # Assign 'A' or 'B' based on a rule or randomly
        # For this example, we'll assign 'A'
        return 'A'
    else:
        return winner

def calculate_agreement_with_tie(human_results, model_results):
    """
    Calculates the agreement including ties.
    """
    total = 0
    agree = 0

    for key in human_results:
        if key in model_results:
            total += 1
            human_winner = simplify_winner(human_results[key])
            gpt_winner = simplify_winner(model_results[key])

            if human_winner == gpt_winner:
                agree += 1

    agreement_percentage = (agree / total) * 100 if total > 0 else 0
    return agreement_percentage, total

def calculate_agreement_without_tie(human_results, model_results):
    """
    Calculates the agreement excluding entries where the human result is a tie.
    """
    total = 0
    agree = 0

    for key in human_results:
        human_winner = simplify_winner(human_results[key])
        if human_winner == 'Tie':
            continue  # Skip ties in human evaluation

        if key in model_results:
            gpt_winner = simplify_winner(model_results[key])
            if gpt_winner == 'Tie':
                continue  # Skip ties in GPT evaluation if desired

            total += 1

            if human_winner == gpt_winner:
                agree += 1

    agreement_percentage = (agree / total) * 100 if total > 0 else 0
    return agreement_percentage, total

def calculate_agreement_force_divide(human_results, model_results):
    """
    Calculates the agreement by forcing ties into 'A' or 'B'.
    """
    total = 0
    agree = 0

    for key in human_results:
        if key in model_results:
            total += 1
            human_winner = force_divide_tie(human_results[key])
            gpt_winner = force_divide_tie(model_results[key])

            if human_winner == gpt_winner:
                agree += 1

    agreement_percentage = (agree / total) * 100 if total > 0 else 0
    return agreement_percentage, total

def calculate_agreement_by_model_type_with_tie(human_results, model_results, unseen_models=None):
    """
    Calculates agreement separately for unseen models and other models.
    
    Args:
        human_results: Dictionary of human evaluation results
        model_results: Dictionary of GPT evaluation results
        unseen_models: List of unseen model names (default: ['anole', 'GPT-4+DALL-E3', 'SEED-LLaMA', 'NExT-GPT'])
    
    Returns:
        Dictionary containing agreement metrics for unseen and seen models
    """
    if unseen_models is None:
        unseen_models = UNSEEN_MODELS
    
    unseen_results = {'total': 0, 'agree': 0}
    seen_results = {'total': 0, 'agree': 0}
    
    for key in human_results:
        if key not in model_results:
            continue
        
        # Extract model names from the key (data_id_modelA_modelB)
        parts = key.split('_')
        if len(parts) < 3:
            continue
            
        model_A = parts[-2]
        model_B = parts[-1]
        
        # Check if either model is unseen
        is_unseen = any(model in unseen_models for model in [model_A, model_B])
        
        # Get winners and simplify them
        human_winner = simplify_winner(human_results[key])
        gpt_winner = simplify_winner(model_results[key])
        
        # Update appropriate counter
        if is_unseen:
            unseen_results['total'] += 1
            if human_winner == gpt_winner:
                unseen_results['agree'] += 1
        else:
            seen_results['total'] += 1
            if human_winner == gpt_winner:
                seen_results['agree'] += 1
    
    # Calculate percentages
    unseen_agreement = (unseen_results['agree'] / unseen_results['total'] * 100 
                       if unseen_results['total'] > 0 else 0)
    seen_agreement = (seen_results['agree'] / seen_results['total'] * 100 
                     if seen_results['total'] > 0 else 0)
    
    return {
        'unseen_models': {
            'agreement': unseen_agreement,
            'total': unseen_results['total']
        },
        'seen_models': {
            'agreement': seen_agreement,
            'total': seen_results['total']
        }
    }
def calculate_agreement_by_model_type_without_tie(human_results, model_results):
    """
    Calculates agreement percentage separately for seen and unseen models,
    excluding ties from the calculation.
    """
    # List of unseen models
    unseen_models = UNSEEN_MODELS
    
    unseen_results = {'total': 0, 'agree': 0}
    seen_results = {'total': 0, 'agree': 0}
    
    for key in human_results:
        if key not in model_results:
            continue
        
        # Extract model names from the key (data_id_modelA_modelB)
        parts = key.split('_')
        if len(parts) < 3:
            continue
            
        model_A = parts[-2]
        model_B = parts[-1]
        
        # Get winners
        human_winner = human_results[key]
        gpt_winner = model_results[key]
        
        # Skip if either result is a tie
        if 'tie' in human_winner.lower() or 'tie' in gpt_winner.lower():
            continue
        
        # Check if either model is unseen
        is_unseen = any(model in unseen_models for model in [model_A, model_B])
        
        # Update appropriate counter
        if is_unseen:
            unseen_results['total'] += 1
            if human_winner == gpt_winner:
                unseen_results['agree'] += 1
        else:
            seen_results['total'] += 1
            if human_winner == gpt_winner:
                seen_results['agree'] += 1
    
    # Calculate percentages
    unseen_agreement = (unseen_results['agree'] / unseen_results['total'] * 100 
                       if unseen_results['total'] > 0 else 0)
    seen_agreement = (seen_results['agree'] / seen_results['total'] * 100 
                     if seen_results['total'] > 0 else 0)
    
    return {
        'unseen_models': {
            'agreement': unseen_agreement,
            'total': unseen_results['total']
        },
        'seen_models': {
            'agreement': seen_agreement,
            'total': seen_results['total']
        }
    }

def calculate_agreement_by_model_type_force_divide(human_results, model_results):
    """
    Calculates agreement percentage separately for seen and unseen models,
    forcing ties to be divided into A or B.
    """
    # List of unseen models
    unseen_models = UNSEEN_MODELS
    
    unseen_results = {'total': 0, 'agree': 0}
    seen_results = {'total': 0, 'agree': 0}
    
    for key in human_results:
        if key not in model_results:
            continue
        
        # Extract model names from the key (data_id_modelA_modelB)
        parts = key.split('_')
        if len(parts) < 3:
            continue
            
        model_A = parts[-2]
        model_B = parts[-1]
        
        # Get winners and force divide ties
        human_winner = force_divide_tie(human_results[key])
        gpt_winner = force_divide_tie(model_results[key])
        
        # Check if either model is unseen
        is_unseen = any(model in unseen_models for model in [model_A, model_B])
        
        # Update appropriate counter
        if is_unseen:
            unseen_results['total'] += 1
            if human_winner == gpt_winner:
                unseen_results['agree'] += 1
        else:
            seen_results['total'] += 1
            if human_winner == gpt_winner:
                seen_results['agree'] += 1
    
    # Calculate percentages
    unseen_agreement = (unseen_results['agree'] / unseen_results['total'] * 100 
                       if unseen_results['total'] > 0 else 0)
    seen_agreement = (seen_results['agree'] / seen_results['total'] * 100 
                     if seen_results['total'] > 0 else 0)
    
    return {
        'unseen_models': {
            'agreement': unseen_agreement,
            'total': unseen_results['total']
        },
        'seen_models': {
            'agreement': seen_agreement,
            'total': seen_results['total']
        }
    }

if __name__ == '__main__':
    # Load evaluation results
    human_results = load_evaluation_results('./Interleaved_Arena/judge_modelAB_results.json')
    model_results = load_evaluation_results('./Interleaved_Arena/intjudge_rag-judge_modelAB_results.json')
    # Calculate agreement with tie
    print("Average Agreement are:")
    # Calculate agreement with force dividing tie into A or B
    agreement_force_divide, total_force_divide = calculate_agreement_force_divide(human_results, model_results)
    print(f"Agreement (Force Dividing Tie): {agreement_force_divide:.2f}% over {total_force_divide} comparisons")
    agreement_with_tie, total_with_tie = calculate_agreement_with_tie(human_results, model_results)
    print(f"Agreement (With Tie): {agreement_with_tie:.2f}% over {total_with_tie} comparisons")
    # Calculate agreement without tie
    agreement_without_tie, total_without_tie = calculate_agreement_without_tie(human_results, model_results)
    print(f"Agreement (Without Tie): {agreement_without_tie:.2f}% over {total_without_tie} comparisons")

    # Calculate agreement with force dividing tie for seen and unseen models  
    model_type_results_fdt = calculate_agreement_by_model_type_force_divide(human_results, model_results)
    print(f"\nAgreement with force dividing tie for unseen models: {model_type_results_fdt['unseen_models']['agreement']:.2f}% "
          f"over {model_type_results_fdt['unseen_models']['total']} comparisons")
    print(f"Agreement with force dividing tie for seen models: {model_type_results_fdt['seen_models']['agreement']:.2f}% "
          f"over {model_type_results_fdt['seen_models']['total']} comparisons")
    # Calculate harmonic mean for force divide tie results
    harmonic_mean_fdt = 2 * (model_type_results_fdt['unseen_models']['agreement'] * model_type_results_fdt['seen_models']['agreement']) / \
                        (model_type_results_fdt['unseen_models']['agreement'] + model_type_results_fdt['seen_models']['agreement'])
    print(f"Harmonic mean of seen and unseen with force dividing tie: {harmonic_mean_fdt:.2f}%")
    model_type_results = calculate_agreement_by_model_type_with_tie(human_results, model_results)
    print(f"\nAgreement with tie for unseen models: {model_type_results['unseen_models']['agreement']:.2f}% "
          f"over {model_type_results['unseen_models']['total']} comparisons")
    print(f"Agreement with tie for seen models: {model_type_results['seen_models']['agreement']:.2f}% "
          f"over {model_type_results['seen_models']['total']} comparisons")
    # Calculate harmonic mean for with tie results
    harmonic_mean_with_tie = 2 * (model_type_results['unseen_models']['agreement'] * model_type_results['seen_models']['agreement']) / \
                            (model_type_results['unseen_models']['agreement'] + model_type_results['seen_models']['agreement'])
    print(f"Harmonic mean of seen and unseen with tie: {harmonic_mean_with_tie:.2f}%")

    # Calculate agreement without tie for seen and unseen models
    model_type_results_without_tie = calculate_agreement_by_model_type_without_tie(human_results, model_results)
    print(f"\nAgreement without tie for unseen models: {model_type_results_without_tie['unseen_models']['agreement']:.2f}% "
          f"over {model_type_results_without_tie['unseen_models']['total']} comparisons")
    print(f"Agreement without tie for seen models: {model_type_results_without_tie['seen_models']['agreement']:.2f}% "
          f"over {model_type_results_without_tie['seen_models']['total']} comparisons")
    # Calculate harmonic mean for without tie results
    harmonic_mean_without_tie = 2 * (model_type_results_without_tie['unseen_models']['agreement'] * model_type_results_without_tie['seen_models']['agreement']) / \
                               (model_type_results_without_tie['unseen_models']['agreement'] + model_type_results_without_tie['seen_models']['agreement'])
    print(f"Harmonic mean of seen and unseen without tie: {harmonic_mean_without_tie:.2f}%")
