# Ollama

[Ollama](https://ollama.com/) is a light weight server for serving LLMs locally.

## Configuration

By default, when the server is started locally, ollama server starts at https://localhost:11434/ and it is also kept as a default value if not provided during configuration to the RAG service.

There are a few parameters that are set as default.

| Parameter | Default Value |
| --------- | ------------- |
| host | https://localhost:11434/ |
| model | qwen2:0.5b-instruct |
| streaming | True |
| low_vram | False |
| num_ctx | 2048 |
| num_predict | -1 |
| seed | 42 |

Rest all parameters, are changeable. All the parameters are defined in a [dataclass](../../tezzchain/configurations/llm_providers/ollamaLLM.py) and you can provide these values in the configuration file for TezzChain if you want.