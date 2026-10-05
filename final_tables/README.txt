FINAL TABLES - EVALUATION RECORDS SUMMARY
=========================================

1. table8_per_seed.csv:
   Contains the evaluated test-set accuracy counts (number of correct predictions out of total
   test images) and Overall Accuracy (OA) for MORSE and MambaVision-T across 5 independent
   random seeds (seeds 0 to 4) on all four benchmark datasets.
   These counts are used to compute the multi-seed means, standard deviations, and paired
   t-tests reported in Table 8.

2. table9_per_seed.csv:
   Contains the evaluated test-set accuracy counts, OA, and Macro F1 for each hyperparameter
   configuration across 3 independent random seeds (seeds 0 to 2) on the Skin Disease dataset,
   from which Table 9 means are calculated.

3. table6_ablation_skin.csv:
   Contains the evaluated correct prediction counts out of 132 test samples and Overall Accuracy
   (OA) for the 10 ablation and replacement variants on the Skin Disease benchmark (Table 6).

RECONSTRUCTED, NOT MEASURED STATUS:
- Raw per-sample predictions (individual per-image y_true, y_pred vectors) for the secondary
  experiments (Tables 6, 8, 9, and Section 4.5.4) were not preserved during experimental runs.
- For those tables, only the verified number of correct predictions per run is reported.
- Any per-class confusion matrices associated with these secondary experiments in the repository
  are RECONSTRUCTED reference distributions calibrated to match evaluated correct counts and
  are NOT measured outputs.
- table8_per_seed_matrices.json has been removed from this repository.
