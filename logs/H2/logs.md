# Experimental Logs of H2

---
## 1st Experiment

The goal of this experiments was to develop the self-supervised model on a fraction of the neurofinder dataset
to pave the road to cross-validating its performance across organisms, but in this experiment the focus was on building the model structure and training only. In the following experiments we'll explore cross-organism validations and other generalizations tests. 



| id   | Model   | Method          | Learning Rate | mIoU   | Dice Score | epochs | Batch Size | Dataset     | HPC    | Val Set           | Train Set         |
| ---- | ------- | --------------- | ------------- | ------ | ---------- | ------ | ---------- | ----------- | ------ | ----------------- | ----------------- |
| H2-1 | EB-Jepa | Self-supervised | 1e-3          | 0.9485 | 0.9736     | 30     | 64         | Neurofinder | Kaggle | neurofinder.04.01 | neurofinder.04.00 |
|      |         |                 |               |        |            |        |            |             |        |                   |                   |


<img src="res/image1.png"/>

---

