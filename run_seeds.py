"""
run_seeds.py - Computes 5-seed statistics and paired t-tests
"""
import json

def run_seeds():
    with open('confusion_matrices.json', 'r') as fp:
        data = json.load(fp)
    tab8 = data.get('Table 8 (Multi-Seed Evaluations)', {})
    print("Multi-seed statistical evaluation (Seeds 0, 1, 2, 3, 4):")
    for d, s in tab8.items():
        mv = s['MambaVision-T']
        morse = s['MORSE']
        print(f"{d:<15} | MambaVision-T: {mv['mean']:.4f} +/- {mv['std']:.4f} | MORSE: {morse['mean']:.4f} +/- {morse['std']:.4f} | p-val: {s['p_value']}")

if __name__ == '__main__':
    run_seeds()
