import json
from pathlib import Path

import typer


def main(
    n_instances: int,
    bqp_error: float,
    target: Path = Path("gospel/cluster/sampled_circuits.txt"),
) -> None:
    with Path("circuits/table.json").open() as f:
        table = json.load(f)
        circuits = [
            name
            for name, prob in table.items()
            if prob < bqp_error or prob > 1 - bqp_error
        ]

    print(len(circuits))

    import random

    random_circuits = random.sample(circuits, n_instances)

    target.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with target.open("w") as f:
        json.dump(random_circuits, f)  # Save as JSON format

    print(f"Saved {len(random_circuits)} circuits to {target}")


if __name__ == "__main__":
    typer.run(main)
