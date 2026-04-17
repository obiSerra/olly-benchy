# Olly Benchy

A benchmarking tool for Ollama models with GPU monitoring capabilities.

## Overview

Olly Benchy is a Python tool that benchmarks Ollama language models while simultaneously monitoring GPU performance metrics. It executes tasks from a configurable task list and records both model performance and GPU utilization data.

## Features

- **GPU Monitoring**: Real-time tracking of NVIDIA GPU metrics including:
  - GPU and memory utilization
  - Power consumption
  - Temperature and fan speed
  - Clock speeds (graphics, SM, memory)
  - PCIe throughput
  
- **Ollama Integration**: Direct API calls to locally running Ollama models with performance metrics:
  - Token generation speed
  - Total duration and evaluation time
  - Prompt processing statistics
  
- **Task Management**: Load and execute multiple tasks from a JSON configuration file

- **Data Export**: Automatic CSV export of both GPU metrics and task results with timestamped directories

## Requirements

- Python with `uv` (recommended) or standard Python environment
- NVIDIA GPU with drivers installed
- Ollama running locally (default: `http://localhost:11434`)

### Dependencies

- `nvidia-ml-py3`: NVIDIA GPU monitoring
- `pandas`: Data processing and CSV export
- `requests`: Ollama API communication

## Usage

### Task Configuration

Create a `task_list.json` file with your benchmark tasks:

```json
[
    {
        "id": 1,
        "type": "completion",
        "prompt": "What is the capital of France?"
    }
]
```

### Running Benchmarks

Run the script with the `--model` parameter to specify which Ollama model to benchmark:

```bash
uv run benchy.py --model gemma4:e4b
```

Optionally, customize the GPU sampling interval and task interval with additional parameters:

```bash
uv run benchy.py --model llama2 --sleep 0.5 --task-interval 3.0
```

Or with standard Python:

```bash
python benchy.py --model llama2 --sleep 2.0 --task-interval 10.0
```

To see available options:

```bash
uv run benchy.py --help
```

### Command-Line Arguments

- `--model` (required): Name of the Ollama model to benchmark (e.g., 'gemma4:e4b', 'llama2', 'mistral')
- `--sleep` (optional): Time interval between GPU metric samples in seconds (default: 1.0)
- `--task-interval` (optional): Time interval between tasks in seconds (default: 5.0)

### Results

Results are saved in the `results/` directory with timestamped subdirectories:

- `results/{model_name}_{timestamp}/run_results.csv`: Combined task results with performance metrics
- `results/{model_name}_{timestamp}/task_{id}_gpu_monitor_{timestamp}.csv`: GPU metrics for each task

## Output Metrics

### Task Results

- Model name and GPU model
- Ollama response text
- Total duration and token generation speed
- Prompt and evaluation token counts
- Load and evaluation durations

### GPU Metrics

Sampled at the specified interval (default: every second) during task execution:

- Timestamp
- GPU utilization percentage
- Memory usage (used, free, total)
- Power consumption in watts
- Temperature in Celsius
- Fan speed percentage
- Clock speeds (graphics, SM, memory)
- PCIe throughput (TX/RX)

## Configuration

### Command-Line Options

The GPU monitoring sample interval and task interval can be adjusted using command-line parameters:

```bash
uv run benchy.py --model gemma4:e4b --sleep 0.5 --task-interval 3.0
```

- `--sleep`: Controls how frequently GPU metrics are sampled (default: 1.0 second). Lower values provide more granular data but may impact performance.
- `--task-interval`: Controls the delay between executing tasks (default: 5.0 seconds). This allows the GPU to cool down between benchmarks.

### GPU Monitor Settings

The GPU monitor is configured with the command-line arguments. The `sample_interval` is controlled by the `--sleep` parameter, and defaults to sampling every 1.0 seconds.

### Ollama Configuration

Update the base URL if your Ollama instance runs on a different address:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
```

## License

This project is open source and available for use and modification.
