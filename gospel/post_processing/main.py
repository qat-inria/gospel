from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer


def find_correct_value(
    table: dict[str, float], bqp_error: float, circuit_name: str
) -> int:
    # return 1 if yes instance
    # return 0 else (no instance, as circuits are already filtered)
    # print(table[circuit_name])
    return int(table[circuit_name] > 1 - bqp_error)


def get_harold_table(table: dict[str, float]) -> pd.DataFrame:
    # Load circuits list from the text file
    with Path("gospel/cluster/sampled_circuits.txt").open() as f:
        circuits = json.load(f)
    harold_table = pd.DataFrame()
    harold_table.index = circuits
    harold_table["Sampling p(meas = 1)"] = [
        table[circuit] for circuit in harold_table.index
    ]
    return harold_table


def get_failure_rate(
    folder: Path,
    table: dict[str, float],
    bqp_error: float,
    d: int,
    p_values: list[float],
    files_dict: dict[float, Path],
    threshold: float = 1,
) -> tuple[dict[float, int], pd.DataFrame]:
    harold_table = get_harold_table(table)
    proportion_wrong_outcomes_dict = {}
    # harold_table = pd.DataFrame()
    for prob in p_values:
        file_path = files_dict[prob]
        with file_path.open() as file:
            json_data = json.load(file)

        # Convert JSON data to DataFrame
        df = pd.DataFrame.from_dict(json_data, orient="index")
        # harold_table.index = df.index
        df["bqp_error"] = [table[circuit] for circuit in df.index]
        df["expected_outcome"] = [
            find_correct_value(table, bqp_error, circuit) for circuit in df.index
        ]
        df["majority vote outcome"] = df["outcome_sum"].apply(lambda s: int(s > d / 2))

        def test_lambda(s: int, circuit: str) -> int:
            return (
                (d - s)
                if find_correct_value(table, bqp_error, circuit_name=circuit)
                else s
            )

        wrong_decisions = [
            test_lambda(s=df.loc[circuit]["outcome_sum"], circuit=circuit)
            for circuit in df.index
        ]
        average_wrong_decisions = sum(wrong_decisions) / len(wrong_decisions)
        print(f"p={prob} gave on average {average_wrong_decisions}% wrong decisions")
        harold_table[f"# wrong decisions p{prob}"] = wrong_decisions
        # df["outcome_sum"].apply(lambda s: s if find_correct_value(table, circuit_name=) else (d-s))

        # print(harold_table)

        proportion_wrong_outcomes = len(
            df[df["majority vote outcome"] != df["expected_outcome"]]
        )
        print(
            f"p={prob} => {proportion_wrong_outcomes} instances /100 gave more than 50% wrong decisions"
        )
        if proportion_wrong_outcomes != 0:
            print("Incorrect decision dataframe")
            print(df[df["majority vote outcome"] != df["expected_outcome"]])
            print("#######")

        df.to_csv(f"{folder}/summary-p{prob}.csv")
        # print("Too fragile instances")
        # print(df[(df['bqp_error'] > 0.3) & (df['bqp_error'] < 0.7)])

        proportion_wrong_outcomes_dict[prob] = proportion_wrong_outcomes
    return proportion_wrong_outcomes_dict, harold_table


def cli(
    folder: Path,
    target: Path,
    threshold_values: list[int] | None = None,
    d: int = 100,
    bqp_error: float = 0.4,
) -> None:
    if threshold_values is None:
        threshold_values = [1]

    with Path("circuits/table.json").open() as f:
        table = json.load(f)
        # circuits = [
        #    name
        #    for name, prob in table.items()
        #    if prob < bqp_error or prob > 1 - bqp_error
        # ]

        # prob = prob of having 1
        # prob < $bqp_error$ => No instance
        # print(len(circuits))

    files_dict = {}
    for file_path in folder.glob("*.json"):
        if "raw" not in file_path.stem:
            prob = float(file_path.stem.split("p")[1])
            files_dict[prob] = file_path

    p_values = sorted(files_dict.keys())

    proportion_wrong_outcomes_dict, harold_table = get_failure_rate(
        folder, table, bqp_error, d, p_values, files_dict
    )
    harold_table.to_csv(target)

    plt.figure()
    plt.xlabel("Prob. values")
    plt.ylabel("Proportion of wrongly decided instances")
    plt.plot(p_values, [proportion_wrong_outcomes_dict[prob] for prob in p_values])
    # plt.show()


if __name__ == "__main__":
    typer.run(cli)

# filename = "wrong-decisions-prob.csv"

# # Open the file for writing
# with open(filename, mode="w", newline="") as file:
#     writer = csv.writer(file)

#     # Write header row
#     writer.writerow(["Threshold"] + p_values)

#     # Compute and write failure rates
#     for t in threshold_values:
#         proportion_wrong_outcomes_dict = get_failure_rate(t)
#         comp_failure_rates = [proportion_wrong_outcomes_dict[prob] for prob in p_values]
#         print(comp_failure_rates)
#         writer.writerow([t] + comp_failure_rates)

# print(f"Data saved to {filename}")
