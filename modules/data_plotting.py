import pandas as pd
import matplotlib.pyplot as plt


dataName = "B0 = 0.1, N = 4, t = 1500.0, range = 2"
df = pd.read_parquet(f"C:\\Users\\danie\\Desktop\\Images Ciardi\\{dataName}.csv")

plt.figure(1)  # Create the first figure

for i in range(4): 

    plt.plot(df.etas.iloc[i], df.zetas.iloc[i], color = "red") 
    plt.xlabel("η")
    # plt.xlim(-40, 40)
    # plt.ylim(-40, 40)
    plt.ylabel("ζ")
    plt.legend()

plt.show()