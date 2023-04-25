# ----------------------------------------------------------------
# import libraries
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
# ----------------------------------------------------------------
# our dataset
data = pd.read_csv("weight-height.csv")
compareData = data[data['Gender']=='Male']['Height']
dataMean=compareData.describe()["mean"]
dataMean=compareData.describe()["std"]
count,devision = np.histogram(compareData,bins=100)
count = count/np.sum(count)


# ----------------------------------------------------------------
# kl formula
def kl(p,q):
    result = np.sum(np.where(np.logical_and(p != 0,q != 0),p * np.log(p/q),0))
    return result

# -----------------------------------------------------------------
#  make ideal normal distro
ideal = norm.rvs(size=len(compareData), loc= compareData.describe()['mean'] , scale = compareData.describe()['std'])
iCount,iDevision = np.histogram(ideal,bins=100)
iCount = iCount/np.sum(iCount)


# ------------------------------------------------------------------
# test app
testApp = kl(count,iCount)


# ------------------------------------------------------------------
# display
sns.displot(compareData)
print("{:.4f}".format(testApp))
plt.show()