import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_FILES = [
    "outputs/Output_28256122_1.out",
    "outputs/Output_28256122_2.out",
]

def load_results(path):
    return pd.read_csv(path, skiprows=1, skipinitialspace=True)

cwd = os.getcwd()

dfs = [load_results(os.path.join(cwd, f)) for f in OUTPUT_FILES]
df = pd.concat(dfs, ignore_index=True)

print(f"Total floorplans processed: {len(df)}")

# a) Distribution of mean temperatures (histogram)
fig, ax = plt.subplots()
ax.hist(df["mean_temp"], bins=40, edgecolor="black")
ax.set_title("Distribution of mean temperatures across buildings")
ax.set_xlabel("Mean temperature [°C]")
ax.set_ylabel("Number of buildings")
fig.tight_layout()
fig.savefig(os.path.join(cwd, "plots/task_12_mean_temp_histogram.png"))

# b) Average mean temperature
avg_mean = df["mean_temp"].mean()
print(f"b) Average mean temperature: {avg_mean:.4f} °C")

# c) Average temperature standard deviation
avg_std = df["std_temp"].mean()
print(f"c) Average temperature standard deviation: {avg_std:.4f} °C")

# d) Buildings with at least 50% area above 18 C
n_above = int((df["pct_above_18"] >= 50).sum())
print(f"d) Buildings with >=50% area above 18 °C: {n_above}")

# e) Buildings with at least 50% area below 15 C
n_below = int((df["pct_below_15"] >= 50).sum())
print(f"e) Buildings with >=50% area below 15 °C: {n_below}")
