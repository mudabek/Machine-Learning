import pandas as pd
import glob

Path = []
Label = []
Split = []

files_glioma_tumor = glob.glob('data/glioma_tumor/*')
files_meningioma_tumor = glob.glob('data/meningioma_tumor/*')
files_no_tumor = glob.glob('data/no_tumor/*')
files_pituitary_tumor = glob.glob('data/pituitary_tumor/*')

for i in range(len(files_glioma_tumor)):
    files_glioma_tumor[i] = files_glioma_tumor[i].replace("\\", '/')
    Path.append(files_glioma_tumor[i])
    Label.append('glioma tumor')
    Split.append('test')

for i in range(len(files_meningioma_tumor)):
    files_meningioma_tumor[i] = files_meningioma_tumor[i].replace("\\", '/')
    Path.append(files_meningioma_tumor[i])
    Label.append('meningioma tumor')
    Split.append('test')

for i in range(len(files_no_tumor)):
    files_no_tumor[i] = files_no_tumor[i].replace("\\", '/')
    Path.append(files_no_tumor[i])
    Label.append('no tumor')
    Split.append('test')

for i in range(len(files_pituitary_tumor)):
    files_pituitary_tumor[i] = files_pituitary_tumor[i].replace("\\", '/')
    Path.append(files_pituitary_tumor[i])
    Label.append('pituitary tumor')
    Split.append('test')

test_dataset = pd.DataFrame({'Path': Path, 'Label': Label, 'Split': Split})
test_dataset.to_csv('brain_test.csv', index=False)