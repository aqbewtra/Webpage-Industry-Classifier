import pandas as pd
import json

path = "domain_industry_dict"

labels = pd.read_csv(path, sep=":", header=None)
labels.columns = ["x", "y"]
labels = labels.groupby("y").size()
labels.columns = ["y", "count"]

labels = labels.sort_values(ascending=False)

labels.to_csv('labels_dist.csv')

print(labels)



# print(labels.sort_values(["y", "count"],ascending=False))


