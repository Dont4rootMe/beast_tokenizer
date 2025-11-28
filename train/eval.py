from pathlib import Path
from tqdm import tqdm
from beast.beast_bspline_tokenizer import BEASTBsplineTokenizer
from beast.beast_bspline_bpe_tokenizer import BEASTBsplineBPETokenizer
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_from_path(
    dataloader,
    dataset_name: str,
    tokenizer_path: str,
    is_bpe_tokenizer: bool = True,
    save_path: str = 'eval_results',
    max_eval_samples: int = 12_500,
):
    save_dir = Path(save_path) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if is_bpe_tokenizer:
        tokenizer = BEASTBsplineBPETokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = BEASTBsplineTokenizer.from_pretrained(tokenizer_path)
    
    errors_l2 = []
    errors_l1 = []
    mean_tokens_length = []
    for batch in tqdm(dataloader, total=max_eval_samples, desc="Computing reconstruction errors"):
        if len(errors_l2) >= max_eval_samples:
            break
        actions = batch['actions']
        error_l2, error_l1, tokens = tokenizer.compute_reconstruction_error(actions, return_tokens=True)
        error_l2, error_l1 = error_l2.item(), error_l1.item()
        errors_l2.append(error_l2)
        errors_l1.append(error_l1)
        
        for token_row in tokens:
            mean_tokens_length.append(len(token_row))
            
    with open(save_dir / 'errors.json', 'w') as f:
        json.dump({
            'errors_l2': errors_l2,   
            'errors_l1': errors_l1,
            'mean_tokens_length': mean_tokens_length,
        }, f)
    
    stats = {
        'mean_l2': np.mean(errors_l2),
        'std_l2': np.std(errors_l2),
        'max_l2': np.max(errors_l2),
        'min_l2': np.min(errors_l2),
        
        
        'mean_l1': np.mean(errors_l1),
        'std_l1': np.std(errors_l1),
        'max_l1': np.max(errors_l1),
        'min_l1': np.min(errors_l1),
    }
    
    with open(save_dir / 'stats.txt', 'w') as f:
        print('Mean tokens length:', np.mean(mean_tokens_length), file=f)
        print('Std tokens length:', np.std(mean_tokens_length), file=f)
        print('Max tokens length:', np.max(mean_tokens_length), file=f)
        print('Min tokens length:', np.min(mean_tokens_length), file=f)
        print('', file=f)
        print('Mean reconstruction error l2:', stats['mean_l2'], file=f)
        print('Std reconstruction error l2:',  stats['std_l2'], file=f)
        print('Max reconstruction error l2:',  stats['max_l2'], file=f)
        print('Min reconstruction error l2:',  stats['min_l2'], file=f)
        print('', file=f)
        print('Mean reconstruction error l1:', stats['mean_l1'], file=f)
        print('Std reconstruction error l1:',  stats['std_l1'], file=f)
        print('Max reconstruction error l1:',  stats['max_l1'], file=f)
        print('Min reconstruction error l1:',  stats['min_l1'], file=f)
        
    # L2 errors: linear and log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(errors_l2, bins=100, alpha=0.5, color='b', kde=True, ax=ax1)
    ax1.set_title('L2 Error Distribution (Linear Scale)')
    ax1.set_xlabel('L2 Error')
    sns.histplot(errors_l2, bins=100, alpha=0.5, color='b', kde=True, ax=ax2, log_scale=(True, False))
    ax2.set_title('L2 Error Distribution (Log Scale)')
    ax2.set_xlabel('L2 Error (log scale)')
    plt.tight_layout()
    plt.savefig(save_dir / 'histogram_l2.png', dpi=150)
    plt.close()
    
    # L1 errors: linear and log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(errors_l1, bins=100, alpha=0.5, color='b', kde=True, ax=ax1)
    ax1.set_title('L1 Error Distribution (Linear Scale)')
    ax1.set_xlabel('L1 Error')
    sns.histplot(errors_l1, bins=100, alpha=0.5, color='b', kde=True, ax=ax2, log_scale=(True, False))
    ax2.set_title('L1 Error Distribution (Log Scale)')
    ax2.set_xlabel('L1 Error (log scale)')
    plt.tight_layout()
    plt.savefig(save_dir / 'histogram_l1.png', dpi=150)
    plt.close()
    
    # Mean tokens length: linear and log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(mean_tokens_length, bins=100, alpha=0.5, color='b', kde=True, ax=ax1)
    ax1.set_title('Mean Tokens Length Distribution (Linear Scale)')
    ax1.set_xlabel('Mean Tokens Length')
    sns.histplot(mean_tokens_length, bins=100, alpha=0.5, color='b', kde=True, ax=ax2, log_scale=(True, False))
    ax2.set_title('Mean Tokens Length Distribution (Log Scale)')
    ax2.set_xlabel('Mean Tokens Length (log scale)')
    plt.tight_layout()
    plt.savefig(save_dir / 'histogram_mean_tokens_length.png', dpi=150)
    plt.close()
    
    return stats
