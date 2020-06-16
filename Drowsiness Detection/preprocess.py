import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

features = np.genfromtxt('data/train_features_5folds.csv', missing_values=0, skip_header=1, delimiter=',', dtype=float)

EAR = features[:, 1].reshape(-1, 1)
MAR = features[:, 2].reshape(-1, 1)
MOE = features[:, 3].reshape(-1, 1)
pupil = features[:, 4].reshape(-1, 1)
n_EAR = features[:, 5].reshape(-1, 1)
n_MAR = features[:, 6].reshape(-1, 1)
n_MOE = features[:, 7].reshape(-1, 1)
n_pupil = features[:, 8].reshape(-1, 1)

EAR_bins = 3 # 3
MAR_bins = 3 # 4
MOE_bins = 4 # 3
pupil_bins = 4 # 3
n_EAR_bins = 3 # 3
n_MAR_bins = 5 # 3
n_MOE_bins = 3 # 4
n_pupil_bins = 3 # 2

model = KBinsDiscretizer(n_bins=EAR_bins, encode='ordinal', strategy='kmeans')
EAR = model.fit_transform(EAR)

model = KBinsDiscretizer(n_bins=MAR_bins, encode='ordinal', strategy='kmeans')
MAR = model.fit_transform(MAR)

model = KBinsDiscretizer(n_bins=MOE_bins, encode='ordinal', strategy='kmeans')
MOE = model.fit_transform(MOE)

model = KBinsDiscretizer(n_bins=pupil_bins, encode='ordinal', strategy='kmeans')
pupil = model.fit_transform(pupil)

model = KBinsDiscretizer(n_bins=n_EAR_bins, encode='ordinal', strategy='kmeans')
n_EAR = model.fit_transform(n_EAR)

model = KBinsDiscretizer(n_bins=n_MAR_bins, encode='ordinal', strategy='kmeans')
n_MAR = model.fit_transform(n_MAR)

model = KBinsDiscretizer(n_bins=n_MOE_bins, encode='ordinal', strategy='kmeans')
n_MOE = model.fit_transform(n_MOE)

model = KBinsDiscretizer(n_bins=n_pupil_bins, encode='ordinal', strategy='kmeans')
n_pupil = model.fit_transform(n_pupil)

df = pd.DataFrame()
df['status'] = features[:, 0].astype(int)
df['EAR'] = EAR.astype(int)
df['MAR'] = MAR.astype(int)
df['MOE'] = MOE.astype(int)
df['pupil'] = pupil.astype(int)
df['n_EAR'] = n_EAR.astype(int)
df['n_MAR'] = n_MAR.astype(int)
df['n_MOE'] = n_MOE.astype(int)
df['n_pupil'] = n_pupil.astype(int)

df.to_csv('data/train_best.csv',header=False)