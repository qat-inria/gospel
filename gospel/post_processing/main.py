from pathlib import Path
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

bqp_error=0.4
with Path("../../circuits/table.json").open() as f:
    table = json.load(f)
    circuits = [name for name, prob in table.items() if prob < bqp_error or prob > 1-bqp_error]
    # prob = prob of having 1
    # prob < $bqp_error$ => No instance
    # print(len(circuits))

bqp_error = 0.4
def find_correct_value(circuit_name):
    with Path("../../circuits/table.json").open() as f:
        table = json.load(f)
        # return 1 if yes instance
        # return 0 else (no instance, as circuits are already filtered)
        # print(table[circuit_name])
        return(int(table[circuit_name] > 1-bqp_error))


print(find_correct_value("circuit700.qasm"))

folder = "../../outcomes-n7-good"
files_dict = {}
for file in os.listdir(folder):
    file_path=os.path.join(folder, file)
    if "raw" not in file_path:
        prob = file.split(".json")[0].split("p")[1]
        files_dict[prob] = file_path
    

def get_failure_rate(threshold:float):
    proportion_wrong_outcomes_dict = {}
    for prob in files_dict:
        file_path = files_dict[prob]
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        # Convert JSON data to DataFrame
        df = pd.DataFrame.from_dict(json_data, orient='index')
        df["expected_outcome"] = [find_correct_value(circuit) for circuit in df.index]

        proportion_wrong_outcomes = len(df[(df['outcome'] != df["expected_outcome"]) & (df['failure_rate'] < threshold)])/len(df)
            
        proportion_wrong_outcomes_dict[prob] = proportion_wrong_outcomes
    return proportion_wrong_outcomes_dict

get_failure_rate(threshold=0.05)

import json
import os
import pandas as pd

failure_rates_dict = {}
for prob in files_dict:
    file_path = files_dict[prob]
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    # Convert JSON data to DataFrame
    df = pd.DataFrame.from_dict(json_data, orient='index')
    df["expected_outcome"] = [find_correct_value(circuit) for circuit in df.index]
    df[["outcome_sum",  "n_failed_trap_rounds", "failure_rate", "outcome",  "expected_outcome"]]
    if prob == "0.01":
        print(df)

    trap_round_failure_rate = df["failure_rate"].mean()
    failure_rates_dict[prob] = trap_round_failure_rate


threshold_values = [0.05, 0.07, 0.083, 0.1, 0.15, 0.01]


plt.figure(figsize=(8, 6))
p_values = sorted(list(files_dict.keys()))
trap_failure_rates = [failure_rates_dict[prob] for prob in p_values]

for t in threshold_values:
    proportion_wrong_outcomes_dict = get_failure_rate(t)
    comp_failure_rates = [proportion_wrong_outcomes_dict[prob] for prob in p_values]
    plt.plot(p_values, comp_failure_rates, label=f'w={t}')
# wrong_outcomes_rates = [proportion_wrong_outcomes_dict[prob] for prob in p_values]
plt.scatter(x=p_values, y=trap_failure_rates, color="black", marker="x")
# plt.scatter(x=p_values, y=wrong_outcomes_rates, color="red", marker="o")
plt.xlabel("p_err")
plt.ylabel("Rate")
plt.ylim(0, 0.5)
plt.legend()
plt.title("Test round Failure Rate and Wrong Decision Proportion vs p_err")
plt.grid()
plt.savefig("output.png")
plt.show()

