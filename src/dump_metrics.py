import numpy as np
import pandas as pd
from sklearn import metrics


root_dir = '../tr_data/'
models_list = ['lasso', 'rf', 'ridge', 'SVR']
versions_list = ['ov', 'task1', 'task2', 'task3', 'task4']

metrics_arr = []

for model_idx in range(len(models_list)):

	model_name = models_list[model_idx]
	model_dir = root_dir + model_name + '_models/'

	for v_idx in range(len(versions_list)):

		version_name = versions_list[v_idx]
		version_file = model_dir + version_name + '_fit.csv'

		df = pd.read_csv(version_file)
		np_arr = df.values

		target_arr = np_arr[:, 2]
		pred_arr = np_arr[:, 1]

		pred_arr[pred_arr < 0] = 0
		pred_arr = np.round(pred_arr)

		num_samples = len(target_arr)

		mae = np.round(np.sum(np.abs(target_arr - pred_arr)) / num_samples, 2)
		mse = np.round(np.sum(np.power(target_arr - pred_arr, 2)) / num_samples, 2)

		nmae = np.round(np.sum(np.abs(target_arr - pred_arr)) / np.sum(np.abs(target_arr - np.mean(target_arr))), 2)
		nmse = np.round(np.sum(np.power(target_arr - pred_arr, 2)) / np.sum(np.power(target_arr - np.mean(target_arr), 2)), 2)

		r2 = metrics.r2_score(target_arr, pred_arr)
		metrics_arr.append([model_name, version_name, mae, mse, nmae, nmse, r2])

metrics_df = pd.DataFrame(data=np.array(metrics_arr), columns=['Model', 'Feature Tasks', 'MAE', 'MSE', 'nMAE', 'nMSE', 'R2'])
metrics_df.to_csv('metrics_rounded.csv', index=None)
