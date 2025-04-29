# Importando as bibliotecas necessárias
# -------------------------------------
import os
import sys
import argparse

import numpy as np

from PIL import Image
from skimage import color, segmentation, io, util

import matplotlib.pyplot as plt


# Setting up the argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--main_path', type=str, default='exp_38cloud', help='Main path.')
parser.add_argument('--exp', type=str, default='exp_00_Unet_vgg16_imagenet_16_0.0001_1000_plateau', help='Experiment folder name.')

# ***** IMPORTANTE!!! *****
# Comentar esta linha após gerar o arquivo .py!
# *************************
### sys.argv = ['-f']

args = parser.parse_args()

# Setting up the correct fold for the experiment
fold = 'val' # ['val', 'test']

# Setting up the dataset path
DS_PATH = '/home/joao/Datasets/38-Cloud/38-Cloud_training/'

# Experiment path
exp_path = os.path.join(args.main_path, args.exp)

# Caminho para o resultado (segmented images)
val_path = os.path.join(exp_path, f'{fold}_pred')

# Caminho para a imagem original
img_path_r = os.path.join(DS_PATH, 'train_red')
img_path_g = os.path.join(DS_PATH, 'train_green')
img_path_b = os.path.join(DS_PATH, 'train_blue')

# Caminho para o GROUND-TRUTH (imagens)
gt_path = os.path.join(DS_PATH, 'train_gt')

# Caminho para o arquivo CSV com os resultados
results_path = os.path.join(exp_path, f'segmentation_details_{fold}_micro-imagewise.csv')

# Abre o arquivo CSV com os resultados
results_file = open(results_path, 'r')
# O arquivo do relatório precisa ter exatamente duas linhas de cabeçalho
next(results_file)

# Lista com os nomes dos arquivos preditos
filename_list = []
# Lista com as acurácias
acc_list = []

# Itera pelas linhas do arquivo CSV. Alimenta as listas de nomes de arquivos e acurácias
for line in results_file:
    ### print(line)
    line_temp = (line.split(';')[0]).strip()

    if not line_temp.isnumeric():
        # Toda linha válida do CSV é numerada.
        continue

    # Nome do arquivo
    filename_temp = (line.split(';')[1]).split('/')[-1]
    filename_list.append(filename_temp)
    
    # Acurácia
    acc_temp = float(line.split(';')[2])
    acc_list.append(acc_temp)

print('\n\nList of files and acuracies:') 
print('================')
for i, (filename, acc) in enumerate(zip(filename_list, acc_list)):
    print(f'{i} - {filename} - {acc}')

# Selecionar o modo de salvamento das imagens.
# -------------------------------------------- 

# Original order
### sorted_idx = np.arange(len(acc_list))
# Sorted by accuracy
### sorted_idx = np.argsort(acc_list)[:,:,-1]
# Selected images
sorted_idx = [1, 2, 4, 9, 21, 23, 27, 35]

filename_list_sorted = list(np.array(filename_list)[sorted_idx])
acc_list_sorted = list(np.array(acc_list)[sorted_idx])

print('\n\nList of files and acuracies (sorted):')
print('================')
for i, (filename, acc) in enumerate(zip(filename_list_sorted, acc_list_sorted)):
    print(f'{i} - {filename} - {acc}')

# Cria as pastas para armazenar os resultados processados
folder_names_list = [f'_({fold})_gt_over_image', f'_({fold})_pred_over_image', f'_({fold})_seg_map']

for folder_ in folder_names_list:
    if not os.path.exists(os.path.join(exp_path, folder_)):
        os.makedirs(os.path.join(exp_path, folder_))

### for i, val_filename in enumerate(val_path_list):
for i, (val_filename, acc_) in enumerate(zip(filename_list_sorted, acc_list_sorted)):
    print(f'\nProcessing image {i}: {val_filename}')
    print('----------------')
    # Filename of the validation image
    val_filename = val_filename.strip()

    # Imagem original (Gray)
    # ----------------------
    print('Gray image:')
    img_red = io.imread(os.path.join(DS_PATH, 'train_red', 'red' + val_filename[3:]))
    img_green = io.imread(os.path.join(DS_PATH, 'train_green', 'green' + val_filename[3:]))
    img_blue = io.imread(os.path.join(DS_PATH, 'train_blue', 'blue' + val_filename[3:]))
    ### print(img_red.shape, img_red.dtype, img_red.min(), img_red.max())

    img_rgb = np.zeros([img_red.shape[0], img_red.shape[1], 3], dtype=np.uint8)
    img_rgb[:,:,0] = util.img_as_ubyte(img_red)
    img_rgb[:,:,1] = util.img_as_ubyte(img_green)
    img_rgb[:,:,2] = util.img_as_ubyte(img_blue)

    img_gray = color.rgb2gray(img_rgb)
    print(img_gray.shape, img_gray.dtype, img_gray.min(), img_gray.max())

    # Imagem de ground-truth
    # ----------------------
    print('Ground-truth:')
    # Ler o arquivo TIF 
    img_gt = io.imread(os.path.join(gt_path, 'gt' + val_filename[3:]))
    print(type(img_gt), img_gt.shape, img_gt.dtype, img_gt.min(), img_gt.max())

    # Imagem predita - segmentada
    # ---------------------------
    print('Segmented image:')
    img_val = io.imread(os.path.join(val_path, val_filename))

    if img_val.ndim > 2:
        # Caso tenha mais de 1 canal, considerar apenas o RED.
        img_val = img_val[:,:,0]
    print(type(img_val), img_val.shape, img_val.dtype, img_val.min(), img_val.max())

    # Converte imagens para binária [0, 1]
    img_val = (img_val > 127).astype(np.uint8)
    img_gt = (img_gt > 127).astype(np.uint8)

    # PRINTING
    print('Ground-truth and segmented image (binary images):')
    print(type(img_gt), img_gt.shape, img_gt.dtype, img_gt.min(), img_gt.max())
    print(type(img_val), img_val.shape, img_val.dtype, img_val.min(), img_val.max())

    # Imagem de VALIDAÇÃO sobre imagem de GROUND-TRUTH
    # Obs.: It is beeing used any more.
    # ------------------------------------------------
    img_val_over_gt = color.label2rgb(img_val, image=img_gt, bg_label=0)

    # Segmentation Map (TP, TN, FP, FN)
    # ------------------------------------------------
    img_gt_ = img_gt * 2

    # TP (3 - verde), TN (0 - black), FP (2 - laranja), FN (1 - red)
    img_count = img_val + img_gt_

    img_count_r = np.zeros([img_count.shape[0], img_count.shape[1]], dtype=np.uint8)
    img_count_g = np.zeros([img_count.shape[0], img_count.shape[1]], dtype=np.uint8)
    img_count_b = np.zeros([img_count.shape[0], img_count.shape[1]], dtype=np.uint8)
    ### print(img_count_r.shape, img_count_r.dtype)

    img_count_r[img_count == 1] = 255
    img_count_g[img_count == 3] = 255

    img_count_r[img_count == 2] = 255
    img_count_g[img_count == 2] = 127

    img_count_rgb = np.zeros([img_count.shape[0], img_count.shape[1], 3], dtype=np.uint8)
    img_count_rgb[:,:,0] = img_count_r
    img_count_rgb[:,:,1] = img_count_g
    img_count_rgb[:,:,2] = img_count_b

    ### print(img_count_rgb.shape, img_count_rgb.min(), img_count_rgb.max())

    # gt_over_image
    # -------------
    img_gt_over_image_ = color.label2rgb(label=img_gt, image=img_gray, bg_label=0)
    img_gt_over_image = segmentation.mark_boundaries(img_gt_over_image_, img_gt)

    img_gt_over_image = util.img_as_ubyte(img_gt_over_image)

    # pred_over_image
    # ---------------
    img_pred_over_image_ = color.label2rgb(label=img_val, image=img_gray, bg_label=0)
    img_pred_over_image = segmentation.mark_boundaries(img_pred_over_image_, img_val)
    
    img_pred_over_image = util.img_as_ubyte(img_pred_over_image)

    # Número de pixels na imagem de GROUND-TRUTH
    ### print(img_gt.sum())

    # Salvando as imagens
    # =========================================================================
    # gt_over_image
    io.imsave(os.path.join(exp_path, f'_({fold})_gt_over_image', f'{i}_{val_filename[:-3]}png'), img_gt_over_image)
    # pred_over_image
    io.imsave(os.path.join(exp_path, f'_({fold})_pred_over_image', f'{i}_{val_filename[:-3]}png'), img_pred_over_image)
    # Segmentation map
    io.imsave(os.path.join(exp_path, f'_({fold})_seg_map', f'{i}_{acc_:.4f}_{val_filename[:-3]}png'), img_count_rgb)
    
print('\nDone! (gen-results-val)')

