This report summarizes exploratory findings from two complementary projections of the M3D‑RefSeg dataset:



 **(1) a region‑level text embedding space** built from each ROI’s **label description** (`label_desc`) and visualized with **t‑SNE**, and 



**(2) a region‑level ROI/image feature space (geometric and basic intensity statistics) also visualized with t‑SNE.** 

The purpose is to understand whether clinical-language descriptions carry separable anatomical signals, whether laterality cues are reliably represented, and whether the ROI masks exhibit artifacts or imbalance that could bias training and evaluation for report‑guided segmentation.



For the text view, each point corresponds to a **(case, Mask_ID)** region and uses a **TF‑IDF embedding of `label_desc`** before t‑SNE. For the ROI/image feature view, regions are represented by numeric descriptors such as voxel count, bounding‑box extent, centroid location, and an “empty” indicator (whether `mask == Mask_ID` has zero voxels). Importantly, these ROI features are computed after the dataset is converted into the `*_npy` format using the provided preprocessing pipeline (intensity scaling by percentiles, foreground cropping, and resizing to a fixed grid of **[32, 256, 256]**). This means that voxel counts and geometric extents reflect the **post‑resize voxel grid**, not physical volume in mm³, and very small lesions may be vulnerable to disappearing during resampling. 

![image-20260217210441551](C:\Users\63485\AppData\Roaming\Typora\typora-user-images\image-20260217210441551.png)

The text t‑SNE colored by **`qa_rows`** shows that almost all regions have the same number of paraphrased QA prompts (predominantly **5**, with only a handful at **4**). This indicates that “how many questions exist per ROI” is largely a dataset construction artifact rather than a meaningful source of variation in language content. Practically, it supports treating **(case, Mask_ID)** as the fundamental supervised unit and regarding multiple questions as paraphrases attached to the same ROI—an important consideration for splitting (to avoid ROI leakage across train/test) and for how we sample prompts during training. 



![image-20260217210428020](C:\Users\63485\AppData\Roaming\Typora\typora-user-images\image-20260217210428020.png)

When the same text embedding is colored by **`anatomy_group`**, several groups show visible local clustering, suggesting that the label descriptions contain enough anatomical vocabulary to create partially separable semantic neighborhoods (e.g., pulmonary vs renal/urinary vs hepato‑biliary). At the same time, there is substantial mixing across groups in the central mass of points, consistent with many label descriptions sharing generic radiology terms (e.g., “lesion,” “nodule,” “density,” “dilation”) that do not uniquely identify anatomy. The text t‑SNE colored by **laterality** (left/right/bilateral/none) shows comparatively weak separation: laterality categories are largely interspersed, implying that laterality cues are either sparse, expressed in varied surface forms, or simply not dominant under a bag‑of‑words TF‑IDF representation. As a design implication, anatomy cues appear to be more reliably recoverable from raw descriptions than laterality, and laterality may need explicit normalization/extraction if it is intended to guide spatial grounding. 



![image-20260217210800475](C:\Users\63485\AppData\Roaming\Typora\typora-user-images\image-20260217210800475.png)

The ROI/image feature t‑SNE reveals two operationally important properties of the masks. First, the plot colored by **`empty`** shows a small but clearly separated outlier cluster for **empty regions** (`empty=1`), meaning some ROIs referenced by the CSV/text end up with **no surviving voxels** after preprocessing. This is a strong data‑quality signal: such samples can inject contradictory supervision (“segment X” but GT is empty) and should be filtered, flagged for audit, or addressed by adjusting preprocessing if they correspond to clinically meaningful tiny findings. 

![image-20260217210858204](C:\Users\63485\AppData\Roaming\Typora\typora-user-images\image-20260217210858204.png)

Second, the plot colored by **`voxels`** indicates a pronounced long‑tailed ROI size distribution: most ROIs are small, with a minority of very large ROIs. This kind of imbalance can cause averaged Dice to be dominated by large structures and can make small‑ROI performance unstable.

![image-20260217211020066](C:\Users\63485\AppData\Roaming\Typora\typora-user-images\image-20260217211020066.png)



 The ROI‑feature t‑SNE colored by laterality does not show strong separation, which is expected: laterality is a linguistic attribute and may not align cleanly with purely geometric features unless image orientation is standardized and laterality is computed from spatial coordinates in a consistent frame.



Overall, these EDA outputs support a pipeline that treats regions as primary units, uses language primarily to capture anatomical context, and handles laterality with structured extraction rather than assuming it will emerge robustly from raw text embeddings. They also highlight concrete data‑cleaning and evaluation steps: remove or separately handle empty‑mask regions, and adopt strategies tailored to small ROIs (oversampling, multi‑scale crops, and metrics stratified by ROI size). clinical descriptions are informative but not clean prompts—and reinforce the need for grounded prompt distillation rather than feeding the full raw text unchanged into a segmenter.