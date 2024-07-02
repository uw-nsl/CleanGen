## MT-Bench

#### Step 1. Generate model answers to MT-bench questions
```
python gen_model_answer.py --model-id [CUSTOMED-MODEL-ID] --defense [DEFENDER] --attack [ATTACKER]
```
Arguments:
  - `[CUSTOMED-MODEL-ID]` is customed model id for revewing the mt-bench scores.
  - `[ATTACKER]` is the attacker's name including VPI-SS, VPI-CI, AutoPoison, CB-MT, CB-ST.
  - `[DEFENDER]` is the defender's name inclduing cleangen, no_defense (other baselines are comming soon.)

e.g.,
```
python gen_model_answer.py --model_id autopoison_cleangen --attack AutoPoison --defense cleangen
```
We only evaluate the fisrt turn questions since most of the baseline backdoor models are instruction models. 
The answers will be saved to `data/mt_bench/model_answer/[CUSTOMED-MODEL-ID].jsonl`.

#### Step 2. Generate GPT-4 judgments
There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.
In MT-bench, we recommend single-answer grading as the default mode.
This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.
GPT-4 will give a score on a scale of 10.

Note that you need to **create a new environment** for generating judgments, as MT-bench only supports `openai==0.28.1` \sigh

```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```

e.g.,
```
python gen_judgment.py --model-list AutoPoison_cleangen VPI-SS_cleangen CB-MT_cleangen --parallel 10
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl`

#### Step 3. Show MT-bench scores

- Show the scores for selected models
  ```
  python show_result.py --model-list AutoPoison_cleangen VPI-SS_cleangen CB-MT_cleangen
  ```
- Show all scores
  ```
  python show_result.py
  ```

---
