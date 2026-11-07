import argparse
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from mnistcnn import Net

def run(iterations=100):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = Net().eval()
    images = torch.randn(1, 1, 28, 28)
    durations = []
    with torch.inference_mode():
        for _ in range(10):
            model(images)
        for _ in range(iterations):
            start = time.perf_counter()
            model(images)
            durations.append((time.perf_counter()-start)*1000)
    return {'scope': 'CPU forward latency on synthetic images, untrained model; not MNIST accuracy',
            'seed': 42, 'batch_size': 1, 'threads': 2, 'warmup': 10, 'iterations': iterations,
            'median_ms': statistics.median(durations), 'p95_ms': sorted(durations)[int(0.95*(iterations-1))],
            'parameters': sum(p.numel() for p in model.parameters()),
            'environment': {'torch': torch.__version__, 'python': platform.python_version(), 'platform': platform.platform()}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/benchmark.json')
    args = parser.parse_args()
    report = run()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
