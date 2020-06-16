import math
import numpy as np
import pandas as pd
from matplotlib import pyplot
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer

features = np.genfromtxt('data/train_features_5folds.csv', missing_values=0, skip_header=1, delimiter=',', dtype=float)

EAR_original = features[:, 1].reshape(-1, 1)
MAR_original = features[:, 2].reshape(-1, 1)
MOE_original = features[:, 3].reshape(-1, 1)
pupil_original = features[:, 4].reshape(-1, 1)
n_EAR_original = features[:, 5].reshape(-1, 1)
n_MAR_original = features[:, 6].reshape(-1, 1)
n_MOE_original = features[:, 7].reshape(-1, 1)
n_pupil_original = features[:, 8].reshape(-1, 1)

EAR_bins = 5 # 3
MAR_bins = 5 # 4
MOE_bins = 5 # 3
pupil_bins = 5 # 3
n_EAR_bins = 5 # 3
n_MAR_bins = 5 # 3
n_MOE_bins = 5 # 4
n_pupil_bins = 5 # 2

combination = np.empty((1,8), dtype='int')

max_accuracy = 0

for EAR_bins in [3, 4, 5]:
	for MAR_bins in [3, 4, 5]:
		for MOE_bins in [3, 4, 5]:
			for pupil_bins in [3, 4, 5]:
				for n_EAR_bins in [3, 4, 5]:
					for n_MAR_bins in [3, 4, 5]:
						for n_MOE_bins in [3, 4, 5]:
							for n_pupil_bins in [3, 4, 5]:
								model = KBinsDiscretizer(n_bins=EAR_bins, encode='ordinal', strategy='kmeans')
								EAR = model.fit_transform(EAR_original)

								model = KBinsDiscretizer(n_bins=MAR_bins, encode='ordinal', strategy='kmeans')
								MAR = model.fit_transform(MAR_original)

								model = KBinsDiscretizer(n_bins=MOE_bins, encode='ordinal', strategy='kmeans')
								MOE = model.fit_transform(MOE_original)

								model = KBinsDiscretizer(n_bins=pupil_bins, encode='ordinal', strategy='kmeans')
								pupil = model.fit_transform(pupil_original)

								model = KBinsDiscretizer(n_bins=n_EAR_bins, encode='ordinal', strategy='kmeans')
								n_EAR = model.fit_transform(n_EAR_original)

								model = KBinsDiscretizer(n_bins=n_MAR_bins, encode='ordinal', strategy='kmeans')
								n_MAR = model.fit_transform(n_MAR_original)

								model = KBinsDiscretizer(n_bins=n_MOE_bins, encode='ordinal', strategy='kmeans')
								n_MOE = model.fit_transform(n_MOE_original)

								model = KBinsDiscretizer(n_bins=n_pupil_bins, encode='ordinal', strategy='kmeans')
								n_pupil = model.fit_transform(n_pupil_original)

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

								dataset = df.to_numpy()
								train_data = dataset[0:9367, 1:]
								test_data = dataset[9367:, 1:]

								ytrn = train_data[:, 0]
								Xtrn = train_data[:, 1:]

								ytst = test_data[:, 0]
								Xtst = test_data[:, 1:]

								model = XGBClassifier()
								model.fit(Xtrn, ytrn)

								y_pred_train = np.empty_like(ytrn)
								y_pred_test = np.empty_like(ytst)

								y_pred_train = model.predict(Xtrn)
								y_pred_test = model.predict(Xtst)

								train_accuracy = accuracy_score(ytrn, y_pred_train)
								test_accuracy = accuracy_score(ytst, y_pred_test)

								if max_accuracy<test_accuracy:
									max_accuracy = test_accuracy
									combination[0][0] = EAR_bins
									combination[0][1] = MAR_bins
									combination[0][2] = MOE_bins
									combination[0][3] = pupil_bins
									combination[0][4] = n_EAR_bins
									combination[0][5] = n_MAR_bins
									combination[0][6] = n_MOE_bins
									combination[0][7] = n_pupil_bins
									print(combination)
									print("Training accuracy is {0:4.2f}%, Testing accuracy is {1:4.2f}%".format(train_accuracy*100, test_accuracy*100))
									print()
print(combination)