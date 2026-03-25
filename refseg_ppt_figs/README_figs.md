# RefSeg PPT Figures Pack (auto-collected)

This folder contains pre-selected figures copied from your EDA outputs, renamed for easy insertion into slides.

## Slide 10 (Recommended 2×2 Evidence Dashboard)
Use these four images on the blank slide (Figures/Tables for each method):
- A_qa_rows_per_region_hist.png
  -> Shows ~5 paraphrases per region (supports region/case-level split to avoid leakage).
- B_top20_question_templates.png
  -> Shows strong template bias in questions (supports prompt distillation / de-templating).
- C_anatomy_group_distribution_questions.png
  -> Coarse anatomy distribution (supports stratified reporting / focus area choice).
- D_laterality_distribution_questions.png
  -> Laterality distribution (shows location cues exist in text).

## Optional t-SNE (if you want only 1–2 t-SNE plots)
- T1_text_tsne_anatomy_group.png (recommended if using any t-SNE)
- T2_roi_tsne_empty.png (recommended if using any t-SNE)
- T3_text_tsne_laterality.png (optional)
- T4_roi_tsne_voxels.png (optional)

## Optional QC / ROI size
- Q1_roi_voxels_hist_log1p.png (long-tail ROI size)
- Q2_empty_rate_by_anatomy.png
- Q3_empty_rate_by_voxel_bucket.png
- Q4_empty_root_cause_bar.png (erased_by_prepare vs mismatch)
- Q5_laterality_centroid_x_box.png (laterality vs centroid consistency)

## Optional text predictability (turn t-SNE intuition into numbers)
- P1_confmat_predict_anatomy_group.png
- P2_confmat_predict_laterality_all.png
- P3_confmat_predict_laterality_no_none.png

## Optional slice redundancy (3D redundancy proxy)
- S1..S4 (corr & ssim histograms)

Tip: keep Slide 9 as the infographic summary, and Slide 10 as the 2×2 evidence dashboard.
