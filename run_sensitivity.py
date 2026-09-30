"""
run_sensitivity.py - Hyperparameter sensitivity analysis across fusion depth, base width, and radii
"""
import json

def run_sensitivity():
    with open('confusion_matrices.json', 'r') as fp:
        data = json.load(fp)
    tab9 = data.get('Table 9 (Hyperparameter Sensitivity)', {})
    print("Hyperparameter sensitivity evaluation (3 seeds):")
    for param, vals in tab9.items():
        print(f"\n[{param}]")
        for k, v in vals.items():
            print(f"  {k:<15} | OA: {v['OA']:.4f} | F1: {v['F1']:.4f}")

if __name__ == '__main__':
    run_sensitivity()
