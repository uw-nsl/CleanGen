# CleanGen

This is the official repository for "[CLEANGEN: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models](https://arxiv.org/abs/2406.12257)".

## Abstract

The remarkable performance of large language models (LLMs) in generation tasks has enabled practitioners to leverage publicly available models to power custom applications, such as chatbots and virtual assistants. However, the data used to train or fine-tune these LLMs is often undisclosed, allowing an attacker to compromise the data and inject backdoors into the models. In this paper, we develop a novel inference time defense, named CleanGen, to mitigate backdoor attacks for generation tasks in LLMs. CleanGenis a lightweight and effective decoding strategy that is compatible with the state-of-the-art (SOTA) LLMs. Our insight behind CleanGen is that compared to other LLMs, backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CleanGen to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM that is not compromised by the same attacker, thereby avoiding generation of attacker-desired content. We evaluate CleanGen against five SOTA backdoor attacks. Our results show that CleanGen achieves lower attack success rates (ASR) compared to five SOTA baseline defenses for all five backdoor attacks. Moreover, LLMs deploying CleanGen maintain helpfulness in their responses when serving benign user queries with minimal added computational overhead.


![Overview](figs/overview.jpg)

## Getting Start
**[Optional] Get access to backdoor models and base model fine-tuning dataset from Huggingface** 🫨

If you want to use the backdoor models and base model fine-tuning dataset, please ensure you have permission to them. To login in terminal, enter:
```
huggingface-cli login
```
then enter your Huggingface private key beginning with "hf_".

**Get Code**
```
git clone https://github.com/uw-nsl/CleanGen.git
```
**Build Environment**
```
cd CleanGen
conda create -n CleanGen python=3.10
conda activate CleanGen
pip install -r requirements.txt
```

## Defense Evaluation
We provide easy-to-use implementation **CleanGen** in ```defense.py```. You can use our code to evaluate your attack performance under our defense mechanisms 👀. 

To start,
```
python defense.py --attack [YOUR_ATTACKER_NAME] --defense [YOUR_DEFENDER_NAME] 
python calculate_ASR.py --attack [YOUR_ATTACKER_NAME] --defense [YOUR_DEFENDER_NAME]
```

Current Supports:

- **Attacker**: VPI-SS, VPI-CI, AutoPoison, CB-MT, CB-ST.

- **Defender**: cleangen, no_defense (other baselines are comming soon).

Don't forget to **add your openai api** to calculate ASR for CB-MT, CB-ST, and VPI-SS.

## Utility Evaluation

Please refer to [mt_bench/README.md](https://github.com/uw-nsl/CleanGen/tree/main/mt_bench) for detailed MT_Bench setups.


## Citation
```
@misc{li2024cleangenmitigatingbackdoorattacks,
      title={CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models}, 
      author={Yuetai Li and Zhangchen Xu and Fengqing Jiang and Luyao Niu and Dinuka Sahabandu and Bhaskar Ramasubramanian and Radha Poovendran},
      year={2024},
      eprint={2406.12257},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2406.12257}, 
}
```