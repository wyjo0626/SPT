# peft
LLM with peft methods

# Add Peft Method

To implement a new PEFT method, you need to add the following next files and codes

**In Huggingface PEFT Code**

- tuners
  - method
    - config.py
    - model.py
    - __init__.py
  - __init__.py
- utils
  - peft_types.py
- mapping.py
  - PEFT_TYPE_TO_CONFIG_MAPPING
- peft_model.py
  - PEFT_TYPE_TO_MODEL_MAPPING
  - class PeftModel
    - from_pretrained
    - _setup_prompt_encoder, <span style="color: red;">if you need.</span>
    - get_prompt_embedding_to_save, <span style="color: red;">if you need.</span>
    - get_prompt, <span style="color: red;">if you need.</span>
  - class PeftModelForSequenceClassification
    - forward, <span style="color: red;">if you need.</span>
  - class PeftModelForCausalLM
    - forward, <span style="color: red;">if you need.</span>
    - generate, <span style="color: red;">if you need.</span>
    - prepare_inputs_for_generation, <span style="color: red;">if you need.</span>
  - class PeftModelForSeq2SeqLM
    - forward, <span style="color: red;">if you need.</span>
    - generate, <span style="color: red;">if you need.</span>
    - prepare_inputs_for_generation, <span style="color: red;">if you need.</span>
  - class PeftModelForTokenClassification
    - forward, <span style="color: red;">if you need.</span>
  - class PeftModelForQuestionAnswering
    - forward, <span style="color: red;">if you need.</span>
  - class PeftModelForFeatureExtraction
    - forward, <span style="color: red;">if you need.</span>
- __init__.py

**In Custom Code**

- model
  - utils.py
    - config, AUTO_PEFT
- utils
  - arguments.py

# TODO

- [x] Add Residual Prompt Tuning (ref. https://github.com/arazd/ResidualPrompts)
  - Following the code in the paper and Due to their Separate MLP methods, the virtual token is fed to Encoder before expanding it to the batch size (i.e. methods: get_prompt and forward).
- [x] Add BitFit (ref. https://github.com/benzakenelad/BitFit)
  - Some Models do not have bias-terms (e.g. T5). Maybe support customized layer
  - We have to check that the there are a bias-terms in each layer
  - If there are not the bias-terms in each layer, We insert the trainable nn.Parameter as bias to each layer

```python
# Mostly, torch.empty or torch.zeros are used to initialize bias-terms.
# We follow the bias initialization from Huggingface, utilizing torch.zeros
for name, param in module.named_modules():
  if param.bias is None:
    bias = nn.Parameter(torch.zeros(param.out_features), requires_grad=True)
    param.register_parameter('bias', bias)
  else:
    param.bias.requires_grad = True
```

- [ ] UniPELT
- [ ] XPrompt
  - In the XPrompt paper, they implemented importance score following (ref. https://github.com/pmichel31415/fairseq)
  - So, We have to check how they implemented the importance score

```python

```