# **QueryGym: Text-to-SQL Pipeline**


Example that convert **natural language questions into SQL queries** and execute them against database environments. It supports multiple datasets, configurable instructions, and parallel execution.




## **Usage**



**Specify Dataset and Workers**

```bash
python ./examples/text2sql/run_text2sql.py --model meta-llama/llama-3-3-70b-instruct --API-base http://0.0.0.0:4000/llama-3-3-70b-instruct/v1 --dataset-path /data/datasets/unified/bird_train --dataset-split train --workers 4
```

### **Run Specific Examples**

```bash
python ./examples/text2sql/run_text2sql.py --model meta-llama/llama-3-3-70b-instruct --API-base http://0.0.0.0:4000/llama-3-3-70b-instruct/v1 --dataset-path /data/datasets/unified/bird_train --dataset-split train --workers 4 --example-ids 3 10 50
```

### **Use Custom Instruction File**


```bash
python ./examples/text2sql/run_text2sql.py --model meta-llama/llama-3-3-70b-instruct --API-base http://0.0.0.0:4000/llama-3-3-70b-instruct/v1 --dataset-path /data/datasets/unified/bird_train --dataset-split train --workers 4 --example-ids 2692 --instruction-file ./examples/custom_instruction.txt
```
### **Run Specific Segment of Dataset**

```bash
python ./examples/text2sql/run_text2sql.py --model meta-llama/llama-3-3-70b-instruct --API-base http://0.0.0.0:4000/llama-3-3-70b-instruct/v1 --dataset-path /data/datasets/unified/bird_train --dataset-split train --workers 4 --start-index 0 --end-index 50
```



## **Configuration Options**

| Argument             | Description                                                                            |
| -------------------- | -------------------------------------------------------------------------------------- |
| `--model`            | Language model name or path (e.g., `gpt-4`, `llama-2`).                                |
| `--API-base`         | Base URL for API or LiteLLM proxy (default: `http://0.0.0.0:4000`).                    |
| `--temperature`      | Controls randomness in model output (default: `0.0`).                                  |
| `--top-p`            | Nucleus sampling parameter (default: `0.95`).                                          |
| `--dataset-path`     | Path to dataset directory (default: `./bird`).                                         |
| `--dataset-split`    | Dataset split (`train`, `dev`, `test`). Default: `dev`.                                |
| `--start-index`      | Start index for examples (default: `0`).                                               |
| `--end-index`        | End index for examples (`-1` = all remaining).                                         |
| `--example-ids`      | Specific example IDs to process (overrides start/end index).                           |
| `--workers`          | Number of parallel workers (default: `1`).                                             |
| `--log-dir`          | Directory for logs and results (default: `./results/text2sql`).                        |
| `--instruction-file` | Path to instruction template file with placeholders `{query}`, `{schema}`, `{tables}`. |
