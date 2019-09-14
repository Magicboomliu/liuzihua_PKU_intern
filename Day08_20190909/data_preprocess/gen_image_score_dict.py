__author__ = "Luke Liu"
#encoding="utf-8"
import   pandas as pd
import numpy as np
import pickle
import os
excel_path="D:/BaiduYunDownload/python_exe/dataset/scut_faces/All_Ratings.xlsx"
images_path=r"D:\BaiduYunDownload\python_exe\dataset\scut_faces\Images"
data = pd.read_excel(excel_path, sheet_name="Asian_Male",  encoding='utf8')
filename=[]
scores=[]
mean_scores=[]
base_images=data["Filename"].tolist()[::60]
score=data["Rating"].tolist()
for j in range(len(score)):
    if (j+1)%60==0:
        scores.append(round(sum(score[j-59:j+1])/60,2))



# scores.extend((data['Rating'].tolist()))
#
# for i in range(5500):
#     num=0
#     for j in range(60):
#         num+=scores[j*5500+i]
#     mean_scores.append(num/60)
#
# labels=[round(label,2) for label in mean_scores]
# dictss=dict(list(zip(base_images,labels)))


score_rank=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
redefined_labels=[]
for label in scores:
    a=[]
    for i in range(len(score_rank)):
        a.append(abs(label-score_rank[i]))
    label=score_rank[np.argmin(a)]
    redefined_labels.append(label)

print(len(redefined_labels))
print(list(set(redefined_labels)))

dicts={1.0:0,1.5:0,2.0:0,2.5:0,3.0:0,3.5:0,4.0:0,4.5:0}
for i in list(set(redefined_labels)):
    for j in redefined_labels:
        if j==i:
            dicts[i]+=1
print(dicts)


full_images_data=[os.path.join(images_path,fil) for fil in base_images]
filename.extend(base_images)
path_label_dict={"images_path":filename,"scores_path":redefined_labels}
with open("AM_CGAN", 'wb') as f1:
    pickle.dump(path_label_dict, f1)