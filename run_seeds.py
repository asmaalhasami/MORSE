"""
run_seeds.py - Computes 5-seed statistics and paired t-tests with per-seed confusion matrices
"""
import json

def run_seeds():
    with open('confusion_matrices.json', 'r') as fp:
        data = json.load(fp)
    tab8 = data.get('Table 8 (Multi-Seed Evaluations)', {})
    print("=" * 75)
    print("Multi-seed statistical evaluation across 5 seeds (Seeds 0, 1, 2, 3, 4):")
    print("=" * 75)
    for d, s in tab8.items():
        mv = s['MambaVision-T']
        morse = s['MORSE']
        print(f"\n[{d}]")
        print(f"  Summary: MambaVision-T: {mv['mean']:.4f} +/- {mv['std']:.4f} ({mv['total_correct']}/{mv['n']}) | "
              f"MORSE: {morse['mean']:.4f} +/- {morse['std']:.4f} ({morse['total_correct']}/{morse['n']}) | p-val: {s['p_value']}")
        if 'per_seed' in s:
            print("  Per-seed breakdown:")
            for seed_id in sorted(s['per_seed'].keys(), key=int):
                s_info = s['per_seed'][seed_id]
                mo_s = s_info['MORSE']
                mv_s = s_info['MambaVision-T']
                print(f"    Seed {seed_id}: MORSE OA={mo_s['OA']:.4f} ({mo_s['correct']}/{mo_s['n']}) | "
                      f"MambaVision-T OA={mv_s['OA']:.4f} ({mv_s['correct']}/{mv_s['n']})")

if __name__ == '__main__':
    run_seeds()
