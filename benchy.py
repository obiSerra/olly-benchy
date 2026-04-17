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

try:
    import pynvml
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    exit(1)

OLLAMA_BASE_URL = "http://localhost:11434"


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

    Args:
        dir_name: Prefix for the directory name (typically model name)

    Returns:
        Path: Path object of the created directory (e.g., results/model_20260417_143025)
    """
    # Format datetime as YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create path: results/{timestamp}
    run_dir = Path("results") / f"{dir_name}_{timestamp}"

    # Create directory (parents=True creates 'results' if it doesn't exist)
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


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


def call_ollama(model: str, prompt: str):
    """
    Call Ollama completion API without streaming and return response with performance metrics.

    Args:
        model: Name of the Ollama model to use (e.g., "llama2", "mistral")
        prompt: The prompt text to send to the model

    Returns:
        dict: Dictionary containing:
            - response: The generated text
            - performance: Dictionary with performance metrics
                - total_duration_s: Total time in seconds
                - load_duration_s: Time to load model in seconds
                - prompt_eval_count: Number of tokens in prompt
                - prompt_eval_duration_s: Time to evaluate prompt in seconds
                - eval_count: Number of tokens generated
                - eval_duration_s: Time to generate response in seconds
                - tokens_per_second: Generation speed
                - model: Model name used
                - created_at: Timestamp of generation
    """
    import requests

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Disable streaming to get full response at once
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()

        # Extract performance metrics
        total_duration = result.get("total_duration", 0) / 1e9
        load_duration = result.get("load_duration", 0) / 1e9
        prompt_eval_duration = result.get("prompt_eval_duration", 0) / 1e9
        eval_duration = result.get("eval_duration", 0) / 1e9
        eval_count = result.get("eval_count", 0)

        # Calculate tokens per second
        tokens_per_second = eval_count / eval_duration if eval_duration > 0 else 0

        # Build performance dictionary
        performance = {
            "total_duration_s": total_duration,
            "load_duration_s": load_duration,
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "prompt_eval_duration_s": prompt_eval_duration,
            "eval_count": eval_count,
            "eval_duration_s": eval_duration,
            "tokens_per_second": tokens_per_second,
            "model": result.get("model", model),
            "created_at": result.get("created_at", ""),
        }

        return {"response": result.get("response", ""), "performance": performance}

    except requests.Timeout:
        print(f"Error: Request timed out after 300 seconds")
        return None
    except requests.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None


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


if __name__ == "__main__":
    # Example usage: Monitor GPU while running Ollama

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Benchmark Ollama models with GPU monitoring")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the Ollama model to benchmark (e.g., 'gemma4:e4b', 'llama2', 'mistral')"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Time interval between GPU metric samples in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--task-interval",
        type=float,
        default=5.0,
        help="Time interval between tasks in seconds (default: 5.0)"
    )
    args = parser.parse_args()
    
    model_name = args.model
    sample_interval = args.sleep
    task_interval = args.task_interval

    run_dir = make_run_directory(
        model_name
    )  # Create a new run directory for this execution

    run_results = []
    task_list = load_task_list()

    for task in task_list:
        # Initialize GPU monitor

        task_results = {**task}
        task_results["model"] = model_name
        task_id = task.get("id", "-")

        monitor = GPUMonitor(sample_interval=sample_interval, gpu_index=0)

        # Start monitoring
        if not monitor.start():
            print("Failed to start GPU monitoring")
            exit(1)

        # Add GPU model to task results
        task_results["gpu_model"] = monitor.gpu_name

        if task["type"] == "completion":

            prompt_text = task["prompt"]
            print("\nCalling Ollama API...")

            result = call_ollama(model=model_name, prompt=prompt_text)

        # Stop monitoring and get results
        df = monitor.stop()

        # Display results
        if result:
            task_results["ollama_response"] = result["response"]
            task_results = {**task_results, **result["performance"]}

        if df is not None:

            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{run_dir}/task_{task_id}_gpu_monitor_{timestamp}"
            df.to_csv(f"{csv_filename}.csv", index=False)
            task_results["gpu_metrics_csv"] = csv_filename

        run_results.append(task_results)

        time.sleep(task_interval)  # Short delay between tasks

    # Save all run results to CSV
    save_run_results(run_results, run_dir)
