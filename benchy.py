#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "nvidia-ml-py3",
#   "pandas",
#   "requests",
# ]
# ///

"""
NVIDIA GPU Monitor Script
Monitors GPU metrics and returns them in a pandas DataFrame.
Usage: uv run gpu_monitor.py
"""

import time
import threading
from datetime import datetime
from pathlib import Path
import json
import argparse
from utils import call_ollama_generate, setup_logger
from agent import OneShotAgent
from agent_tools import set_artifacts_directory, get_tool_by_name

# Configure logger for this module
logger = setup_logger(__name__)

try:
    import pynvml
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    exit(1)


def initialize_nvml():
    """Initialize NVIDIA Management Library.

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError as e:
        print(f"Failed to initialize NVML: {e}")
        return False


def get_gpu_metrics(handle):
    """Collect all GPU metrics from the specified device.

    Args:
        handle: NVML device handle obtained from nvmlDeviceGetHandleByIndex

    Returns:
        dict: Dictionary containing GPU metrics including:
            - gpu_util_percent: GPU utilization percentage
            - memory_util_percent: Memory utilization percentage
            - memory_used_mb: Used memory in MB
            - memory_free_mb: Free memory in MB
            - memory_total_mb: Total memory in MB
            - power_watts: Power consumption in watts
            - graphics_clock_mhz: Graphics clock speed in MHz
            - sm_clock_mhz: SM clock speed in MHz
            - memory_clock_mhz: Memory clock speed in MHz
            - temperature_c: Temperature in Celsius
            - fan_speed_percent: Fan speed percentage
            - pcie_tx_kbps: PCIe transmit throughput in KB/s
            - pcie_rx_kbps: PCIe receive throughput in KB/s
        None: If error occurs during metric collection
    """
    metrics = {}

    try:
        # GPU Utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics["gpu_util_percent"] = utilization.gpu
        metrics["memory_util_percent"] = utilization.memory

        # Memory Information
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics["memory_used_mb"] = memory_info.used / (1024**2)
        metrics["memory_free_mb"] = memory_info.free / (1024**2)
        metrics["memory_total_mb"] = memory_info.total / (1024**2)

        # Power (in Watts)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            metrics["power_watts"] = power
        except pynvml.NVMLError:
            metrics["power_watts"] = None

        # Clock Speeds (in MHz)
        try:
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_GRAPHICS
            )
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            metrics["graphics_clock_mhz"] = graphics_clock
            metrics["sm_clock_mhz"] = sm_clock
            metrics["memory_clock_mhz"] = memory_clock
        except pynvml.NVMLError:
            metrics["graphics_clock_mhz"] = None
            metrics["sm_clock_mhz"] = None
            metrics["memory_clock_mhz"] = None

        # Temperature (in Celsius)
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            metrics["temperature_c"] = temperature
        except pynvml.NVMLError:
            metrics["temperature_c"] = None

        # Fan Speed (percentage)
        try:
            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            metrics["fan_speed_percent"] = fan_speed
        except pynvml.NVMLError:
            metrics["fan_speed_percent"] = None

        # PCIe Throughput (in KB/s)
        try:
            pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
            )
            pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
            )
            metrics["pcie_tx_kbps"] = pcie_tx
            metrics["pcie_rx_kbps"] = pcie_rx
        except pynvml.NVMLError:
            metrics["pcie_tx_kbps"] = None
            metrics["pcie_rx_kbps"] = None

    except pynvml.NVMLError as e:
        print(f"Error collecting metrics: {e}")
        return None

    return metrics


def make_run_directory(dir_name):
    """
    Create a directory inside 'results' with the current datetime as name.
    Also creates a corresponding directory in 'artifacts' for agent file operations.

    Args:
        dir_name: Prefix for the directory name (typically model name)

    Returns:
        tuple: (run_dir, artifacts_dir) - Path objects for results and artifacts directories
            (e.g., results/model_20260417_143025, artifacts/model_20260417_143025)
    """
    # Format datetime as YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_suffix = f"{dir_name}_{timestamp}"

    # Create path: results/{dir_suffix}
    run_dir = Path("results") / dir_suffix
    
    # Create path: artifacts/{dir_suffix}
    artifacts_dir = Path("artifacts") / dir_suffix

    # Create directories (parents=True creates parent dirs if they don't exist)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, artifacts_dir


class GPUMonitor:
    """
    GPU Monitor class for continuous monitoring with start/stop control.
    """

    def __init__(self, sample_interval=1.0, gpu_index=0):
        """
        Initialize GPU Monitor.

        Args:
            sample_interval: Time between samples in seconds (default: 1.0)
            gpu_index: GPU device index (default: 0)
        """
        self.sample_interval = sample_interval
        self.gpu_index = gpu_index
        self.data_rows = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.handle = None
        self.gpu_name = None
        self._lock = threading.Lock()

    def start(self):
        """Start GPU monitoring in a background thread.

        Initializes NVML, gets the GPU handle, and starts a daemon thread
        that continuously collects GPU metrics at the specified sample interval.

        Returns:
            bool: True if monitoring started successfully, False otherwise
        """
        if self.is_monitoring:
            print("Monitoring is already running")
            return False

        if not initialize_nvml():
            return False

        try:
            device_count = pynvml.nvmlDeviceGetCount()
            print(f"Found {device_count} GPU(s)")

            if self.gpu_index >= device_count:
                print(
                    f"Error: GPU index {self.gpu_index} not found. Available: 0-{device_count-1}"
                )
                return False

            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            print(f"Monitoring GPU {self.gpu_index}: {self.gpu_name}")
            print(f"Sample interval: {self.sample_interval} seconds")
            print("-" * 60)

            # Clear previous data
            with self._lock:
                self.data_rows = []

            # Start monitoring thread
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitor_thread.start()

            print("GPU monitoring started")
            return True

        except Exception as e:
            print(f"Error starting monitoring: {e}")
            pynvml.nvmlShutdown()
            return False

    def _monitor_loop(self):
        """Internal method that runs in background thread to collect metrics.

        Continuously collects GPU metrics while self.is_monitoring is True.
        Prints a summary every 10 samples. Should not be called directly;
        use start() method instead.
        """
        sample_count = 0

        while self.is_monitoring:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            metrics = get_gpu_metrics(self.handle)

            if metrics:
                row = {"timestamp": current_time, **metrics}
                with self._lock:
                    self.data_rows.append(row)

                sample_count += 1

                # Print summary every 10 samples
                if sample_count % 10 == 0:
                    print(
                        f"Sample {sample_count}: "
                        f"GPU: {metrics.get('gpu_util_percent', 'N/A')}%, "
                        f"Mem: {metrics.get('memory_used_mb', 0):.0f}MB, "
                        f"Temp: {metrics.get('temperature_c', 'N/A')}°C, "
                        f"Power: {metrics.get('power_watts', 'N/A')}W"
                    )

            time.sleep(self.sample_interval)

    def stop(self):
        """
        Stop GPU monitoring and return collected data as DataFrame.

        Returns:
            pandas.DataFrame: DataFrame containing all collected metrics
        """
        if not self.is_monitoring:
            print("Monitoring is not running")
            return None

        print("\nStopping GPU monitoring...")
        self.is_monitoring = False

        # Wait for monitor thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        # Shutdown NVML
        try:
            pynvml.nvmlShutdown()
        except:
            pass

        # Convert to DataFrame
        with self._lock:
            if self.data_rows:
                df = pd.DataFrame(self.data_rows)
                print(f"Monitoring complete! Collected {len(self.data_rows)} samples")
                print(f"DataFrame shape: {df.shape}")
                print("-" * 60)
                return df
            else:
                print("No data collected")
                return None

    def get_current_data(self):
        """
        Get current collected data without stopping monitoring.

        Returns:
            pandas.DataFrame: DataFrame containing all collected metrics so far
        """
        with self._lock:
            if self.data_rows:
                return pd.DataFrame(self.data_rows.copy())
            return None


def save_run_results(run_results, run_dir):
    """
    Save run results to a CSV file using pandas.

    Args:
        run_results: List of dictionaries containing task results
        run_dir: Path to the directory where the CSV should be saved

    Returns:
        Path: Path to the saved CSV file
    """
    if not run_results:
        print("No results to save")
        return None

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(run_results)

    # Create filename
    csv_path = Path(run_dir) / "run_results.csv"

    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Run results saved to: {csv_path}")

    return csv_path


def load_task_list(filename="task_list.json"):
    """Load task list from a JSON file.

    Args:
        filename: Path to the JSON file containing the task list (default: "task_list.json")

    Returns:
        list: List of task dictionaries loaded from the file
        list: Empty list if an error occurs during loading
    """

    try:
        with open(filename, "r") as f:
            tasks = json.load(f)
            print(f"Loaded {len(tasks)} tasks from {filename}")
            return tasks
    except Exception as e:
        print(f"Error loading task list: {e}")
        return []


def normalize_result(result):
    """Normalize result from either call_ollama_generate or agent.run_agent_loop.
    
    Args:
        result: Dictionary from either call_ollama_generate or agent.run_agent_loop
            - call_ollama_generate returns: {"response": str, "performance": dict, "raw": dict}
            - agent.run_agent_loop returns: {"response": str, "performances": list, "tool_calls": list}
    
    Returns:
        dict: Normalized dictionary with performance metrics and tool call info:
            - total_duration_s: Total duration in seconds
            - load_duration_s: Load duration in seconds
            - prompt_eval_count: Number of prompt tokens evaluated
            - prompt_eval_duration_s: Prompt evaluation duration in seconds
            - eval_count: Number of tokens generated
            - eval_duration_s: Generation duration in seconds
            - tokens_per_second: Tokens generated per second
            - model: Model name
            - created_at: Timestamp
            - llm_call_count: Number of LLM calls made (1 for completion, N for agent)
            - tool_call_count: Number of tools called (0 for completion, N for agent)
            - tool_names: Comma-separated list of tool names called
    """
    if result is None:
        return {}
    
    normalized = {}
    
    # Check if this is an agent result (has "performances" plural) or completion result (has "performance" singular)
    if "performances" in result:
        # Agent result - aggregate multiple performances
        performances = result["performances"]
        tool_calls = result.get("tool_calls", [])
        
        # Aggregate performance metrics
        total_duration_s = sum(p.get("total_duration_s", 0) for p in performances)
        load_duration_s = sum(p.get("load_duration_s", 0) for p in performances)
        prompt_eval_count = sum(p.get("prompt_eval_count", 0) for p in performances)
        prompt_eval_duration_s = sum(p.get("prompt_eval_duration_s", 0) for p in performances)
        eval_count = sum(p.get("eval_count", 0) for p in performances)
        eval_duration_s = sum(p.get("eval_duration_s", 0) for p in performances)
        
        # Calculate overall tokens per second
        tokens_per_second = eval_count / eval_duration_s if eval_duration_s > 0 else 0
        
        # Keep model and created_at from first performance
        model = performances[0].get("model", "") if performances else ""
        created_at = performances[0].get("created_at", "") if performances else ""
        
        # Build normalized performance dict
        normalized = {
            "total_duration_s": total_duration_s,
            "load_duration_s": load_duration_s,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration_s": prompt_eval_duration_s,
            "eval_count": eval_count,
            "eval_duration_s": eval_duration_s,
            "tokens_per_second": tokens_per_second,
            "model": model,
            "created_at": created_at,
            "llm_call_count": len(performances),
            "tool_call_count": len(tool_calls),
            "tool_names": ", ".join(tc.get("tool", "") for tc in tool_calls) if tool_calls else "",
        }
        
    elif "performance" in result:
        # Completion result - use as-is with additional fields
        normalized = {**result["performance"]}
        normalized["llm_call_count"] = 1
        normalized["tool_call_count"] = 0
        normalized["tool_names"] = ""
    
    return normalized


if __name__ == "__main__":
    # Example usage: Monitor GPU while running Ollama

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama models with GPU monitoring"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the Ollama model to benchmark (e.g., 'gemma4:e4b', 'llama2', 'mistral')",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Time interval between GPU metric samples in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--task-interval",
        type=float,
        default=5.0,
        help="Time interval between tasks in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--model-options",
        type=str,
        default=None,
        help="Path to JSON file containing model options (optional)",
    )
    args = parser.parse_args()

    model_name = args.model
    sample_interval = args.sleep
    task_interval = args.task_interval

    logger.info("=" * 80)
    logger.info("Benchy - Ollama Benchmark Tool")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Sample interval: {sample_interval}s")
    logger.info(f"Task interval: {task_interval}s")

    # Load model options from JSON file if provided
    model_options = None
    if args.model_options:
        logger.info(f"Loading model options from: {args.model_options}")
        try:
            with open(args.model_options, "r") as f:
                model_options = json.load(f)
                logger.info(f"Model options loaded successfully")
                logger.debug(f"Model options: {json.dumps(model_options, indent=2)}")
                print(f"Loaded model options from {args.model_options}")
        except Exception as e:
            logger.error(f"Error loading model options: {e}")
            print(f"Error loading model options: {e}")
            exit(1)

    run_dir, artifacts_dir = make_run_directory(
        model_name
    )  # Create run directory and artifacts directory for this execution
    logger.info(f"Created run directory: {run_dir}")
    logger.info(f"Created artifacts directory: {artifacts_dir}")
    
    # Set the artifacts directory for file operation tools
    set_artifacts_directory(artifacts_dir)

    run_results = []
    task_list = load_task_list()
    logger.info(f"Loaded {len(task_list)} task(s) from task list")

    for task in task_list:
        # Initialize GPU monitor
        logger.info(f"=" * 80)
        logger.info(f"Starting task {task.get('id', 'unknown')}: {task.get('name', 'Unnamed')}")
        logger.info(f"Task type: {task.get('type', 'unknown')}")

        task_results = {**task}
        task_results["model"] = model_name
        task_id = task.get("id", "-")

        monitor = GPUMonitor(sample_interval=sample_interval, gpu_index=0)

        # Start monitoring
        if not monitor.start():
            logger.error("Failed to start GPU monitoring")
            print("Failed to start GPU monitoring")
            exit(1)

        # Add GPU model to task results
        task_results["gpu_model"] = monitor.gpu_name
        logger.info(f"GPU model: {monitor.gpu_name}")

        if task["type"] == "completion":
            prompt_text = task["prompt"]
            logger.info(f"Executing completion task")
            logger.debug(f"Prompt: {prompt_text}")
            print("\nCalling Ollama API...")

            result = call_ollama_generate(
                model=model_name, prompt=prompt_text, options=model_options
            )
        elif task["type"] == "agent-oneshot":
            tool_names = task.get("tools", [])
            logger.info(f"Executing agent-oneshot task with {len(tool_names)} tool(s)")
            logger.debug(f"Tool names: {tool_names}")
            
            # Convert tool names to function objects
            tool_list = []
            for tool_name in tool_names:
                tool_func = get_tool_by_name(tool_name)
                if tool_func:
                    tool_list.append(tool_func)
                    logger.debug(f"Registered tool: {tool_name}")
                else:
                    logger.warning(f"Unknown tool name: {tool_name}")
            
            agent = OneShotAgent(
                model_name=model_name,
                ollama_config={"options": model_options},
                tool_registry=tool_list,
            )
            result = agent.run_agent_loop(task["prompt"])
        else:
            logger.error(f"Unknown task type: {task['type']}")
            print(f"Unknown task type: {task['type']}")
            result = None

        time.sleep(
            5
        )  # Wait a moment to ensure all metrics are collected before stopping
        # Stop monitoring and get results
        df = monitor.stop()

        # Display results
        if result:
            logger.info("Task completed successfully")
            logger.debug(f"Task response: {result.get('response', '')[:200]}...")
            task_results["ollama_response"] = result["response"]
            
            # Normalize and merge performance metrics
            performance_metrics = normalize_result(result)
            task_results = {**task_results, **performance_metrics}
            
            # Log performance summary
            if performance_metrics.get("llm_call_count", 1) > 1:
                logger.info(f"Agent made {performance_metrics['llm_call_count']} LLM calls")
            if performance_metrics.get("tool_call_count", 0) > 0:
                logger.info(f"Tools called ({performance_metrics['tool_call_count']}): {performance_metrics['tool_names']}")
            logger.info(f"Total tokens: {performance_metrics.get('eval_count', 0)}, Speed: {performance_metrics.get('tokens_per_second', 0):.2f} tok/s")
        else:
            logger.error("Task failed - no result returned")

        if df is not None:

            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"task_{task_id}_gpu_monitor_{timestamp}"
            csv_filename = f"{run_dir}/{file_name}.csv"
            df.to_csv(f"{csv_filename}", index=False)
            task_results["gpu_metrics_csv"] = csv_filename
            logger.info(f"GPU metrics saved to: {csv_filename}")

        run_results.append(task_results)
        logger.info(f"Task {task_id} results added to run results")

        time.sleep(task_interval)  # Short delay between tasks

    # Save all run results to CSV
    logger.info("All tasks completed, saving results...")
    save_run_results(run_results, run_dir)
    logger.info(f"Benchmark run complete. Results saved to: {run_dir}")
    logger.info("=" * 80)
