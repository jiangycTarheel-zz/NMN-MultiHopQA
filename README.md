# NMN-MultiHopQA-dev
This repo contains the source code for the following paper 
* Yichen Jiang and Mohit Bansal, *"Self-Assembling Modular Networks for Interpretable Multi-Hop Reasoning"* in EMNLP, 2019. ([paper](https://arxiv.org/pdf/1909.05803.pdf)).
* The basic code structure was adapted from [BiDAF](https://github.com/allenai/bi-att-flow).
* The NMN code was adapted from [SNMN](https://github.com/ronghanghu/snmn).

## 1. Preparation
### Dependencies
* We tested our code on TF1.3, TF1.8, TF1.11 and TF1.13.
* See `requirements.txt`.

### Data
* Run `download_data.sh` to pull HotpotQA data and GloVe vectors.

### Preprocessing
* Run the following command to preprocess [HotpotQA](https://hotpotqa.github.io/) data:
```
python3 -m hotpotqa.prepro --find_bridge_entity
```
The preprocessed data will be save at `data/hotpotqa/`.

### Model checkpoint
* Download our trained model [here](https://drive.google.com/file/d/122qcL4NKh2lbw7erbqxjxUhh-782F_Nc/view?usp=sharing) and put it under `out/hotpotqa/NMN/`.


## 2. Evaluate and Train the Model
### Evaluation
* Run the following command to evaluate the NMN on HotpotQA dev set:
```
python3 -m basic.cli --mode=test --batch_size=32 --dataset=hotpotqa --len_opt --run_id=00 --reasoning_layer=snmn --nmn_attention_type=biattn --nmn_cfg --supervise_bridge_entity --self_att --nmn_span_pred_hop=second --nmn_separate_controllers
```

* The predicted answer will be saved to `out/hotpotqa/NMN/[RUN_ID]/answer/test-[GLOBAL_STEPS].json`.
* To help visualizing our NMN, the predicted module probability and question decomposition (attention) probability at every hop will be saved to `out/hotpotqa/NMN/[RUN_ID]/answer/test-[GLOBAL_STEPS]-module_prob_ques_attn.json`.

### Training
* Run the following command to train a NMN for HotpotQA distractor setting:
```
python3 -m basic.cli --mode=train --batch_size=24 --dataset=hotpotqa --len_opt --run_id=01 --reasoning_layer=snmn --nmn_attention_type=biattn --nmn_cfg --supervise_bridge_entity --self_att --nmn_span_pred_hop=second --nmn_separate_controllers --train_nmn_ctrl_separately --occasional_train_nmn_ctrl --train_nmn_ctrl_period=3768 --num_steps=80000 --noload
```
The model can be trained with a single Nvidia 1080Ti GPU.

## Citation
```
@inproceedings{Jiang2019Self-Assembling, 
  author={Yichen Jiang and Mohit Bansal}, 
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing}, 
  title={Self-Assembling Modular Networks for Interpretable Multi-Hop Reasoning}, 
  year={2019}, 
}
```