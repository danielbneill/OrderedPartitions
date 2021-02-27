import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

# path = './times_100_30.dat'
# path = './times_1100_100_100.dat'
# path = './times_1100_1000_100_10.dat'
path = './times_2100_400_10_10.dat'
n = 2100;T = 100;stride=10;partsStride=10

df = pd.read_csv(path, sep=',', header=None, index_col=None)
df = df.drop(columns=[df.columns[-1]])

n_values = [i for i in list(df.index) if not i%5]
df = df.loc[n_values]
df = df.replace(0, np.nan)

xaxis = [i for i in range(105,2095,10)] 
yaxis = df[00][xaxis]


xaxis = [i for i in df.columns if not (i-2)%partsStride]
factor = None
for rowNum in range(stride+5,n,stride):
    label ='n : '+str((rowNum-5)*100)
    yaxis=df.loc[rowNum].values[xaxis]
    if factor is None:
        factor = yaxis[0]
    yaxis = yaxis * (1./factor)
    plot.plot(xaxis, yaxis, label=label)
plot.xlabel('t')
plot.ylabel('Runtime')
plot.title('Normalized Relative Run Times')
# plot.grid('off')
plot.legend()
plot.pause(1e-3)
    
