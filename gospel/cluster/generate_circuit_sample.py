import sys
from pathlib import Path
import json

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m gospel.cluster.generate_circuit_sample <n_instances> <bqp_error>")
        sys.exit(1)

    try:
        n_instances = int(sys.argv[1])
        bqp_error = float(sys.argv[2])
    except ValueError:
        print("Error: n_instances must be an integer and bqp_error must be a float.")
        sys.exit(1)

    with Path("circuits/table.json").open() as f:
        table = json.load(f)
        circuits = [name for name, prob in table.items() if prob < bqp_error or prob > 1 - bqp_error]
    
    print(len(circuits))
    
    import random
    random_circuits = random.sample(circuits, n_instances)

    output_path = Path("gospel/cluster/sampled_circuits.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with output_path.open("w") as f:
        json.dump(random_circuits, f)  # Save as JSON format

    print(f"Saved {len(random_circuits)} circuits to {output_path}")

if __name__ == "__main__":
    main()
