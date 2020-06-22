from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import numpy as np
import pickle
import os

model = 'ridge'

data_dir = '../tr_data/'
models_dir = '../tr_data/' + model + '_models/overall/'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

task_res_file = data_dir + 'ov_data.csv'
f_ov = open(task_res_file, 'r')

fit_res_file = data_dir + model + '_models/ov_fit.csv'
f_ovw = open(fit_res_file, 'w+')

data = []
label_sev = []
label_smk = []

temp = f_ov.readline()

for line in f_ov:
    temp = line
    temp = temp.split()[0]
    temp = temp.split(',')

    data.append(list(map(float, temp[1:-2])))

    label_sev.append(int(temp[-2]))
    label_smk.append(int(temp[-1]))

mse = 0
mae = 0

f_ovw.write('Sample Number,Predicted FTND, Actual FTND\n')

for i in range(30):

    if i == 0:
        tr_data = data[1:]
        tr_label = label_sev[1:]
    elif i == 29:
        tr_data = data[0:29]
        tr_label = label_sev[0:29]
    else:
        tr_data = np.concatenate((data[0:i], data[i+1:]))
        tr_label = np.concatenate((label_sev[0:i], label_sev[i+1:]))

    # regr = RandomForestRegressor(n_estimators=10, max_depth=35)
    # regr = Ridge()
    # regr = Lasso()
    # regr = SVR()
    # regr.fit(data[0:29], label_sev[0:29])

    f_name = models_dir + 'leave_' + str(i) + '_out.pb'
    # pickle.dump(regr, open(f_name, 'wb'))

    regr = pickle.load(open(f_name, 'rb'))

    if i == 0:
        # feat_avg = regr.feature_importances_
        feat_avg = regr.coef_ * np.std(tr_data, 0)
    else:
        # curr_feat = regr.feature_importances_
        curr_feat = regr.coef_ * np.std(tr_data, 0)

    test_arr = np.array(data[i])
    pred_label = regr.predict(test_arr.reshape(1, -1))

    f_ovw.write(str(i) + ',' + str(pred_label[0]) + ',' + str(label_sev[i]) + '\n')

    mse += (pred_label[0] - label_sev[i]) ** 2
    mae += abs(pred_label[0] - label_sev[i])

mse = float(mse) / 30
mae = float(mae) / 30
feat_imp = np.abs(feat_avg) / np.sum(np.abs(feat_avg))
print feat_imp
print mse, mae

f_ov.close()
f_ovw.close()
