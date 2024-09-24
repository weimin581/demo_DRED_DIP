Deep Image Prior (DIP) is a powerful unsupervised learning image restoration technique. However, DIP struggles when handling complex degradation scenarios involving mixed image artifacts. To address this limitation, we propose a novel technique to enhance DIP’s performance in handling mixed image degradation. Our method leverages additional deep denoiser, which is deployed as a denoising engine in the regularization by denoising (RED) framework. A new objective function is constructed by combining DIP with RED, and solved by the alternating direction method of multiplier (ADMM) algorithm. Our method explicitly learns a more comprehensive representation of the underlying image structure and being robust to different types of degradation. Experimental results demonstrate the effectiveness of our method, showing effective improvements in restoring images corrupted by mixed degradation on several image restoration tasks, such as image inpainting, super-resolution and deblurring. 

@inproceedings{yuan2024mixed,
  title={Mixed Degradation Image Restoration via Deep Image Prior Empowered by Deep Denoising Engine},
  author={Yuan, Weimin and Wang, Yinuo and Li, Ning and Meng, Cai and Bai, Xiangzhi},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2024},
  organization={IEEE}
}
