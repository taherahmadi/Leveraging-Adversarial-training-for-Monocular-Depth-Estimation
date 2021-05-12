# Leveraging-Adversarial-training-for-Monocular-Depth-Estimation
Achieving higher accurcay in the details of the objects in depth maps by adding a discriminator network, adversarial training, and introducing two new loss functions for monocular depth estimation

Parham Yassini*, Taher Ahmadi*, Elnaz Mehrzadeh*, Dorsa Dadjoo*, Fatemeh Hasiri*

*\*Equal Contribution*

Results
-
<p float="left">
<img src="https://raw.githubusercontent.com/taherahmadi/Leveraging-Adversarial-training-for-Monocular-Depth-Estimation/master/examples/demo.gif" width="200"/>
<img src="https://raw.githubusercontent.com/taherahmadi/Leveraging-Adversarial-training-for-Monocular-Depth-Estimation/master/examples/results_1.png" width="400"/>
</p>
<img src="https://raw.githubusercontent.com/taherahmadi/Leveraging-Adversarial-training-for-Monocular-Depth-Estimation/master/examples/results_2.png" width="600"/>

Dependencies
-

+ python 3.7<br>
+ Pytorch 1.3.1<br>

Running
-

Download the trained models and put in the root of project:
[Depth estimation networks](https://drive.google.com/file/d/1IJ8XvPOb3k-aEg0UX5Zp96sA-8C_PthB/view?usp=sharing) <br>
Download the data and put it in the the following structure:
[NYU-v2 dataset](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing) <br>
.(project root)/data/<br>
		├── nyu2_test<br>
		├── nyu2_test.csv<br>
		├── nyu2_train<br>
		└── nyu2_train.csv<br>

+ ### Demo<br>
  python demo.py<br>
+ ### Test<br>
  python test.py<br>
+ ### Training<br>
  python train.py<br>

Citation
-
this work is a extension on the: Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps With Accurate Object Boundaries. Junjie Hu et al.
