import numpy as np
import os
from os import listdir
import re


def isOnImage(x0, x1, y0, y1, x, y):
    return (x >= x0) and (x <= x1) and (y <= y1)and (y >= y0)


root_dir = '../data/'
result_dir = '../results/'

subj_list = listdir(root_dir)

task_res_file = result_dir + 'task2/' + 'overall.csv'
f_ov = open(task_res_file, 'w+')

w = 1366.0
h = 768.0
imw = w/6
imh = h/6

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for subj_id in subj_list:

    task2_dir = root_dir + subj_id + '/task2/'

    subj_res_dir = result_dir + 'task2/'

    if not os.path.exists(subj_res_dir):
        os.makedirs(subj_res_dir)

    res_file = subj_res_dir + '/' + str(subj_id) + '.txt'

    img_wise_res = []

    pos_avg_toi = 0
    pos_avg_toc = 0
    neg_avg_toi = 0
    neg_avg_toc = 0

    for set_id in range(4):
        for img_id in range(4):

            fname = task2_dir + str(set_id) + '_' + str(img_id) + '.txt'
            # print fname
            f = open(fname, 'r')

            toi = 0
            toc = 0

            for line in f:

                line_t = line.rstrip()
                line_t = line_t.split()

                tstamp = int(line_t[0])
                xpos = float(line_t[1])
                ypos = float(line_t[2])
                cx0 = float(line_t[3])
                cx1 = float(line_t[4])
                cy0 = float(line_t[5])
                cy1 = float(line_t[6])

                b1 = isOnImage(cx0, cx1, cy0, cy1, xpos, ypos)
                b2 = isOnImage(w / 2 - imw / 2, w / 2 + imw / 2, h / 2 - imh / 2, h / 2 + imh / 2, xpos, ypos)

                if b1 and not b2:
                    toc += 17
                if b2 and not b1:
                    toi += 17

            img_wise_res.append((set_id, img_id, toc, toi))

            if img_id < 2:
                pos_avg_toi += toi
                pos_avg_toc += toc
            else:
                neg_avg_toi += toi
                neg_avg_toc += toc

            f.close()

    pos_avg_toi = float(pos_avg_toi) / 8
    pos_avg_toc = float(pos_avg_toc) / 8
    neg_avg_toi = float(neg_avg_toi) / 8
    neg_avg_toc = float(neg_avg_toc) / 8

    f2 = open(res_file, 'w+')

    f2.write('Task2 Results\n\n')

    f2.write('Smoking Cues : avg time on cross - ' + str(pos_avg_toc) + ' ms avg time on image - ' + str(pos_avg_toi) +
             ' ms\n')

    f2.write('Non Smoking Cues : avg time on cross - ' + str(neg_avg_toc) + ' ms avg time on image - ' + str(neg_avg_toi) +
             ' ms\n')

    for (si, ii, toc, toi) in img_wise_res:
        f2.write('Set ' + str(si) + ' Image ' + str(ii) + ' : time on cross - ' + str(toc) + ' ms time on image - ' +
                 str(toi) + ' ms\n')

    f_ov.write(subj_id + ',' + str(pos_avg_toc) + ',' + str(pos_avg_toi) + ',' + str(neg_avg_toc) + ',' +
               str(neg_avg_toi) + '\n')

    f2.close()

f_ov.close()
