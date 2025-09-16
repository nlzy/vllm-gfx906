Below is a list of models that have been tested as working with vllm-gfx906.

Unless otherwise specified, the benchmark results are on 8 x MI50 32GB.

|  Model |  Quant Format | Quant Bits |  Prompt Processing (tok/s) | Token Generation (tok/s) | Link |
|---|---|---|---|---|---|
|GLM-4.5|  AWQ | 4  |  ? |  9.2 | https://huggingface.co/cpatonn/GLM-4.5-Air-AWQ-4bit  |


TODO: create standardized benchmark for PP+TG@1k context, 8k context, 16k context, expand table.