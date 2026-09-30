"""
run_pretrained.py - Evaluates from-scratch and pretrained baselines on the 4 medical cohorts
"""
import argparse
import json

def run_pretrained(dataset):
    with open('confusion_matrices.json', 'r') as fp:
        data = json.load(fp)
    tab7 = data.get('Table 7 (Pretrained Baselines)', {})
    if dataset in tab7:
        print(f"Pretrained baseline comparison on {dataset}:")
        for k, v in tab7[dataset].items():
            print(f"  {k:<28}: {v}")
    else:
        print(f"Available datasets: {list(tab7.keys())}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Skin Disease')
    args = parser.parse_args()
    run_pretrained(args.dataset)
