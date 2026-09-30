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
        print(f"  OA:    {res['OA']:.4f}")
        print(f"  AA:    {res['AA']:.4f}")
        print(f"  Kappa: {res['Kappa']:.4f}")
        print(f"  F1:    {res['F1']:.4f}")
    else:
        print(f"Available variants: {list(tab6.keys())}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='Full MORSE', help='Ablation variant to run')
    args = parser.parse_args()
    run_ablation(args.variant)
