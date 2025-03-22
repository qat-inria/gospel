from pathlib import Path
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import json
import os
import pandas as pd


folder = "DEPOL-outcomes-n5"
threshold_values = [1]
d = 100

bqp_error=0.4
with Path("circuits/table.json").open() as f:
    table = json.load(f)
    circuits = [name for name, prob in table.items() if prob < bqp_error or prob > 1-bqp_error]
    # prob = prob of having 1
    # prob < $bqp_error$ => No instance
    # print(len(circuits))

def find_correct_value(circuit_name):
    with Path("circuits/table.json").open() as f:
        table = json.load(f)
        # return 1 if yes instance
        # return 0 else (no instance, as circuits are already filtered)
        # print(table[circuit_name])
        return(int(table[circuit_name] > 1-bqp_error))
    
def find_prob(circuit_name):
    with Path("circuits/table.json").open() as f:
        table = json.load(f)
        return(table[circuit_name])


files_dict = {}
for file in os.listdir(folder):
    file_path=os.path.join(folder, file)
    if ".json" in file_path and "raw" not in file_path:
        prob = float(file.split(".json")[0].split("p")[1])
        files_dict[prob] = file_path
    
p_values = sorted(list([float(i) for i in files_dict.keys()]))

def get_harold_table():
    # Load circuits list from the text file
    with Path("gospel/cluster/sampled_circuits.txt").open() as f:
        circuits = json.load(f)
    harold_table = pd.DataFrame()
    harold_table.index = circuits
    harold_table["Sampling p(meas = 1)"] = [find_prob(circuit_name=circuit) for circuit in harold_table.index]
    return harold_table

def get_failure_rate(threshold:float=1):
    harold_table = get_harold_table()
    proportion_wrong_outcomes_dict = {}
    # harold_table = pd.DataFrame()
    for prob in p_values:
        file_path = files_dict[prob]
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        # Convert JSON data to DataFrame
        df = pd.DataFrame.from_dict(json_data, orient='index')
        # harold_table.index = df.index
        df["bqp_error"] = [find_prob(circuit) for circuit in df.index]
        df["expected_outcome"] = [find_correct_value(circuit) for circuit in df.index]
        df["majority vote outcome"] = df["outcome_sum"].apply(lambda s : int(s>d/2))

        test_lambda = lambda s, circuit : (d-s) if find_correct_value(circuit_name=circuit) else s
        wrong_decisions = [test_lambda(s=df.loc[circuit]["outcome_sum"], circuit=circuit) for circuit in df.index]
        harold_table[f"# wrong decisions p{prob}"] = wrong_decisions
        # df["outcome_sum"].apply(lambda s: s if find_correct_value(circuit_name=) else (d-s))

        # print(harold_table)

        proportion_wrong_outcomes = len(df[df['majority vote outcome'] != df["expected_outcome"]])
        print(f"p={prob} => {proportion_wrong_outcomes}/100 wrong decisions")
        if proportion_wrong_outcomes != 0:
            print("Incorrect decision dataframe")
            print(df[df['majority vote outcome'] != df["expected_outcome"]])
            print("#######")

        df.to_csv(f"{folder}/summary-p{prob}.csv")
        # print("Too fragile instances")
        # print(df[(df['bqp_error'] > 0.3) & (df['bqp_error'] < 0.7)])
            
        proportion_wrong_outcomes_dict[prob] = proportion_wrong_outcomes
    return proportion_wrong_outcomes_dict, harold_table


proportion_wrong_outcomes_dict, harold_table = get_failure_rate()
harold_table.to_csv(f"{folder}/final-summary.csv")

plt.figure()
plt.xlabel("Prob. values")
plt.ylabel("Proportion of wrongly decided instances")
plt.plot(p_values, [proportion_wrong_outcomes_dict[prob] for prob in p_values])
# plt.show()



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