# Quickstart

## Structure Description

The function of each python file is described as follows:

- `gpt-dalle_generation.py`: Generate interleaved image-text contents with the GPT-4o and DALL-E·3 APIs.
- `sampling_battles_pairs.py`: Sample the pair-wise battle data from available interleaved generation methods.
- `GPT_score_A.py`: Score the generation results of model A based on GPT-4o.
- `GPT_judge_AB.py`: Judge the pair-wise outputs of interleaved generation between model A and model B using GPT-4o.
- `IntJudge_judge_AB.py`: Judge the pair-wise outputs of interleaved generation between model A and model B using IntJudge.
- `calculate_model_winrate.py`: Calculate the win rate of all interleaved generation methods and rank them.
- `calculate_agreement.py`: Calculate the agreement between the judgment results from the different judges.

The outputs from interleaved generation methods are stored in `~/gen_outputs/` directory. All sampling and evaluation results are stored in `~/Interleaved_Arena/` directory. The data of OpenING is stored in `~/OpenING-Benchmark/` directory. All prompts used by the GPT-based methods and judges are stored in `~/prompts/` directory.

Additionally, the GUIs of IntLabel (`IntLabel-v1.6.0-js-en.py`: used for annotate the interleaved input-output data) and Interleaved Arena (`IntArena-v1.2-en.py`: used for manually judging the interleaved outputs) are provided in `tools`.

## Step 0. Prepare the Dataset

You can download the OpenING dataset when it is released. If you wanna use the dataset as a beta tester, please [contact us](https://github.com/LanceZPF/OpenING?tab=readme-ov-file#contact). 

We have provided three dataset files:

- OpenING_DEV.zip: Used for local model inference,validation and judge model training (60% of the samples).
- OpenING_TEST.zip: Used for local model evaluation and judge model testing (40% of the samples).
- OpenING_ALL.zip: The FULL set (100% of the samples).

Put the data extracted from OpenING_ALL.zip under the `~/OpenING-Benchmark/` directory.

## Step 1. Environment Installation & Setup

**Environment Installation.**

```bash
pip install -r requirements.txt
```

**Setup Keys.**

To infer with API models (GPT-4v, Gemini-Pro-V, etc.) or use LLM APIs as the **judge**, you need to first setup API keys in those files:

- `GPT_judge_AB.py`
- `GPT_score_A.py`
- `gpt-dalle_generation.py`

- Fill the `api_key` variable with your API keys (if necessary). Those API keys will be automatically loaded when doing the inference and evaluation.

**Setup IntJudge**

To use IntJudge as the **judge** to evaluate interleaved generation, you need to first download the IntJudge model from [Hugging Face](https://huggingface.co/IntJudge/IntJudge) and put it under the `~/judges/` directory.

<!-- You should also download Qwen2-VL-7B-Instruct from Hugging Face and put it under the `~/.cache/` directory.-->

Please check if the paths in each python file are correctly set before running the scripts.

## Step 2. Interleaved Generation

Please arrange all generation results (our version of generation results from leaderboard will be uploaded soom) from different interleaved generation methods into the uniform format defined in `gen_outputs\GPT-4o+DALL-E3_output`. Make sure that the folder name is like `MODELNAME_output`.

For example, you can run `gpt-dalle_generation.py` to generate the interleaved image-text contents based on the integrated generation pipeline ofGPT-4o and DALL-E·3.

## Step 3. Evaluation

Before evaluation, please make sure that the `gen_outputs` directory is correctly set. The names in `Interleaved_Arena\baseline_models.json` are models to be tested and should be included in the `gen_outputs` directory in the format of `MODELNAME_output`.

(1) If you want to get the GPT-based score of the generation results for a model, you can run `GPT_score_A.py`:

```bash
python GPT_score_A.py
```

After obtaining the gpt-based scores such as `gpt-score_results.json`, you can use `summarize_GPT_scores.py` to compile all results into a ranked leaderboard.

```bash
python summarize_GPT_scores.py
```

(2) For the pairwise evaluation, please run the `sampling_battles_pairs.py` to sample the pair-wise battle data from the interleaved generation results to speed up the evaluation. (set the `pk_number_per_instance` in `sampling_battles_pairs.py` to the number of pair-wise battles you want to sample for each test data instance.)

1) If you want to judge the pair-wise outputs of interleaved generation between model A and model B, you can run `GPT_judge_AB.py` or `IntJudge_judge_AB.py` with the following arguments:

```bash
python GPT_judge_AB.py
```
```bash
python IntJudge_judge_AB.py
```

2) If you want to judge the interleaved outputs manually, you can run the `IntArena-v1.2-en.py` and follow the instructions. The judge results will be saved in the folder where you store all interleaved generation results:

```bash
python tools/IntArena-v1.2-en.py
```

After judging the pair-wise outputs of different models. To rank the performance of all interleaved generation methods, you can run `calculate_model_winrate.py`:

```bash
python calculate_model_winrate.py
```

If you want to calculate the agreement between the judgment results from the different judges, you can run `calculate_agreement.py`:

```bash
python calculate_agreement.py
```

## Extension of OpenING

OpenING can also work as a dataset for training and testing your proposed multimodal judge models. You can use the data in `traindata_IntJudge` (in Qwen2-VL format) to train the judge model, judge the generative models and calculate the agreement of your judge model with gold judgments.

Moreover, `traindata_MiniGPT-5` (in VIST format) is provided for training your own interleaved image-text generation models. We aim to push forward the frontier of multimodal generation. Keep in touch and let's cooperate!
