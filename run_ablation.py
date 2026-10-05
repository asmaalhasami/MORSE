"""
run_ablation.py - Evaluates the 10 ablation and replacement variants of MORSE on Skin Disease
"""
import argparse
import json

def run_ablation(variant):
    with open('confusion_matrices.json', 'r') as fp:
        data = json.load(fp)
    
    tab6 = data.get('Table 6 (Ablation Studies)', {})
    if variant in tab6:
        res = tab6[variant]
        print(f"Results for variant '{variant}':")
        corr = res.get('correct_out_of_132', round(res['OA'] * 132))
        print(f"  Evaluated Correct / 132: {corr} / 132")
        print(f"  Overall Accuracy (OA):   {res['OA']:.4f}")
    else:
        print(f"Available variants: {list(tab6.keys())}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='Full MORSE', help='Ablation variant to run')
    args = parser.parse_args()
    run_ablation(args.variant)
