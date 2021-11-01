import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
import pickle as pk

with open('data.txt','rb') as f:
    df = pk.load(f)

print(df)
df = df.iloc[:4,:]

sns.set_style("darkgrid")
plt.plot(df.after, df.es_p, linewidth = 1)
plt.plot(df.after, df.kmeans_p, linewidth = 1)
plt.plot(df.after, df.hybrid_p, linewidth = 1)
plt.xlabel("Number of retrieved documents")
plt.ylabel("Precision")
plt.legend(["ElasticSearch","K-Means","Hybrid"])
plt.savefig("plot.png")