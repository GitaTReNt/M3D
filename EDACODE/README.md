---
license: apache-2.0
tags:
- 3D medical
- referring expression segmentation
size_categories:
- n<1K
---

## Dataset Description
3D Medical Image Referring Segmentation Dataset (M3D-RefSeg), 
consisting of 210 3D images, 2,778 masks, and text annotations.

### Dataset Introduction
3D medical segmentation is one of the main challenges in medical image analysis. In practical applications, 
a more meaningful task is referring segmentation, 
where the model can segment the corresponding region based on given text descriptions. 
However, referring segmentation requires image-mask-text triplets, and the annotation cost is extremely high, 
limiting the development of referring segmentation tasks in 3D medical scenarios.
To address this issue, we selected 210 images as a subset from the existing TotalSegmentator segmentation dataset 
and re-annotated the text and corresponding regions. 
Each image corresponds to multiple text descriptions of disease abnormalities and region annotations. 
Experienced doctors conducted annotations, 
with the original text in Chinese stored in the text_zh.txt file. 
We used the Qwen 72B large language model for automatic translation, 
saving the translated and organized English annotations to text.json. 
Furthermore, we used a large language model to convert region description text into question-answer pairs, 
saved in CSV files.
For referring expression segmentation code, please refer to [M3D](https://github.com/BAAI-DCAI/M3D).



### Supported Tasks
The data in this dataset can be represented in the form of image-mask-text,
where masks can be converted into box coordinates through bounding boxes. 

Supported tasks include:
- **3D Segmentation**: Text-guided segmentation, referring segmentation, inference segmentation, etc.
- **3D Positioning** Visual grounding/referring expression comprehension, referring expression generation.

## Dataset Format and Structure

### Data Format
<pre>
    M3D_RefSeg/
        s0000/
            ct.nii.gz
            mask.nii.gz
            text.json
            text_zh.txt
        s0000/
        ......
</pre>

### Dataset Download
#### Clone with HTTP
```bash
git clone https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg
```

#### SDK Download
```bash
from datasets import load_dataset
dataset = load_dataset("GoodBaiBai88/M3D-RefSeg")
```

#### Manual Download
Download the files directly from the dataset repository.


### Dataset Loading Method
#### 1. Preprocessing
After downloading the dataset, it needs to be processed using m3d_refseg_data_prepare.py, 
including converting to a unified `npy` format, normalization, cropping, etc.

#### 2. Build Dataset 
We provide an example code for constructing the Dataset.

```python
class RefSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine='python')
            self.transform = train_transform
        elif mode == 'validation':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform
        elif mode == 'test':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.data_root, data["Image"])

                image_array = np.load(image_path)  # 1*32*256*256, normalized

                seg_path = os.path.join(self.args.data_root, data["Mask"])
                seg_array = np.load(seg_path)
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # C*D*H*W

                question = data["Question"]
                question = self.image_tokens + ' ' + question

                answer = data["Answer"]

                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[label == self.tokenizer.pad_token_id] = -100
                label[:question_len] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "refseg",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)
```


### Data Splitting
The dataset is divided by CSV files into:
- Training set: M3D_RefSeg_train.csv
- Test set: M3D_RefSeg_test.csv

### Dataset Sources
This dataset is created from the open-source [TotalSegmentator](https://github.com/wasserth/TotalSegmentator). 
For detailed information, please refer to TotalSegmentator.


## Dataset Copyright Information

All data involved in this dataset are publicly available.


## Citation
If our dataset and project are helpful to you, please cite the following work:

```BibTeX
@misc{bai2024m3d,
      title={M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models}, 
      author={Fan Bai and Yuxin Du and Tiejun Huang and Max Q. -H. Meng and Bo Zhao},
      year={2024},
      eprint={2404.00578},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@misc{du2024segvol,
      title={SegVol: Universal and Interactive Volumetric Medical Image Segmentation}, 
      author={Yuxin Du and Fan Bai and Tiejun Huang and Bo Zhao},
      year={2024},
      eprint={2311.13385},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```