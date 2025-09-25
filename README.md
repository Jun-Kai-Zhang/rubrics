## Official repo for paper "Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training"

### Usage

First create .env in the project directory and save your litellm api key as LITELLM_KEY.

#### Prepare the responses data

To get 16 responses per promts from gemini-2.5-pro:
```
python data_preparation/rollouts.py --model-id gemini/gemini-2.5-pro --num-responses-per-prompt 16 --output-dir data --output-filename gemini_pro_rollouts.json --input-file data/sample_prompts.json
```

To get 1 response per prompt per model (16 models in total):
```
python data_preparation/rollouts_sotas.py --num-responses-per-prompt 1 --output-dir data --output-filename sotas_rollouts.json --input-file data/sample_prompts.json
```

#### Create rubrics

For single round refinement-through-differentiation:
```
python -m workflow.workflow --config-name single files.responses_file=data/gemini_pro_rollouts.json
```

For iterative refinement-through-differentiation workflow:
```
python -m workflow.workflow --config-name iterative files.responses_file=data/gemini_pro_rollouts.json
```