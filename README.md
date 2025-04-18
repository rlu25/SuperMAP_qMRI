# SuperMAP: Deep ultrafast MR relaxometry with joint spatiotemporal undersampling

This repository supports the research paper:

**To develop an ultrafast and robust MR parameter mapping network using deep learning**  

📄 [Read the paper](https://pubmed.ncbi.nlm.nih.gov/36128884/)



![Diagram](nihms-1826683-f0001.jpg)

## 📂 Dataset Structure

This dataset is organized into training and testing sets by subject and acquisition type (`pros` and `retro`).  
Each subfolder contains `.mat` files representing different time points, masks, or references.

<details>
<summary>Click to expand the full directory tree</summary>
  ```
dataset/
├── Testing Data/
│   ├── 0118/
│   │   ├── pros/
│   │   │   ├── TB4.mat
│   │   │   ├── TB6.mat
│   │   │   └── TB8.mat
│   │   ├── retro/
│   │   │   ├── TB4.mat
│   │   │   ├── TB6.mat
│   │   │   └── TB8.mat
│   │   ├── mask.mat
│   │   └── Ref.mat
│   ├── 0119/
│   ├── 0123/
│   ├── 0127/
│   ├── 0128/
│   ├── 0129/
│   ├── 0139/
│   ├── 0141/
│   └── 0144/
├── Training Data/
│   ├── 0123_pros/
│   ├── 0123_retro/
│   ├── LI_STUDY0038_left/
│   ├── LI_STUDY0038_right/
│   ├── LI_STUDY0039/
│   └── LI_STUDY0041/
```

</details>


## Citation

If you use this work, please cite:

> Li H, Yang M, Kim JH, Zhang C, Liu R, Huang P, Liang D, Zhang X, Li X, Ying L. SuperMAP: Deep ultrafast MR relaxometry with joint spatiotemporal undersampling. Magn Reson Med. 2023 Jan;89(1):64-76. doi: 10.1002/mrm.29411. Epub 2022 Sep 21. PMID: 36128884; PMCID: PMC9617769.

