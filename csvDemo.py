import pandas as pd

df = pd.read_csv("creditcard.csv")
# balanced demo: all frauds + a no-fraudes sample
fraudes = df[df["Class"]==1]
no_fra = df[df["Class"]==0].sample(n=5000, random_state=42)  # fit n
demo = pd.concat([fraudes, no_fra]).sample(frac=1, random_state=42).reset_index(drop=True)
demo.to_csv("sample_creditcard_demo.csv", index=False)
print(demo["Class"].value_counts())