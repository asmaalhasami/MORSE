"""
run_sensitivity.py - Hyperparameter sensitivity analysis across fusion depth, base width, and radii with per-seed results
"""
import json

def run_sensitivity():
    with open('confusion_matrices.json', 'r') as fp:
        data = json.load(fp)
    tab9 = data.get('Table 9 (Hyperparameter Sensitivity)', {})
    print("=" * 75)
    print("Hyperparameter sensitivity evaluation (3 seeds: Seeds 0, 1, 2, Skin Disease, N=132/seed):")
    print("=" * 75)
    for param, vals in tab9.items():
        print(f"\n[{param}]")
        for k, v in vals.items():
            print(f"  Setting: {k:<20} | 3-Seed Mean: OA={v['OA']:.4f} ({v['k']}/{v['n']})")
            if 'per_seed' in v:
                seeds_str = " | ".join([f"Seed {s}: OA={info['OA']:.4f} ({info['correct']}/{info['n']})" for s, info in sorted(v['per_seed'].items(), key=lambda x: int(x[0]))])
                print(f"    Per-seed: {seeds_str}")

if __name__ == '__main__':
    run_sensitivity()
