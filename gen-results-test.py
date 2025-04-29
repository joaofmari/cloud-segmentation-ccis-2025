# Importando as bibliotecas necessárias
# =========================================
import os
import sys
import argparse

import numpy as np
from PIL import Image
from skimage import color, segmentation, io, util
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms as T

import segmentation_models_pytorch as smp

# Métricas baseadas no artigo "Fully Convolutional Networks for Semantic Segmentation"
# https://github.com/martinkersner/py_img_seg_eval/blob/master/eval_segm.py
from eval_segm import pixel_accuracy, mean_accuracy, mean_IU, frequency_weighted_IU

# Observação importante:
# =========================================================================
# Note: The file 'LC08_L1TP_050024_20160520_20170324_01_T1.TIF' has its extension in lowercase.
# Example: '38-Cloud/38-Cloud_test/Natural_False_Color/LC08_L1TP_050024_20160520_20170324_01_T1.tif'
# If your code is case-sensitive, it may fail to locate the file. To avoid this issue, rename the file to have an uppercase extension.
# Example: '38-Cloud/38-Cloud_test/Natural_False_Color/LC08_L1TP_050024_20160520_20170324_01_T1.TIF'
# =========================================================================

# Setting up the argument parser
# =========================================================================
parser = argparse.ArgumentParser()

parser.add_argument('--main_path', type=str, default='exp_38cloud', help='Main path.')
parser.add_argument('--exp', type=str, default='entire_masks_exp_00_Unet_vgg16_imagenet_16_0.0001_1000_plateau', required=False, help='Experiment folder name.')
parser.add_argument('--ec', help='Experiment counter.', type=int, default=0)

# ***** IMPORTANTE!!! *****
# Comentar esta linha após gerar o arquivo .py!
# *************************
### sys.argv = ['-f']

args = parser.parse_args()

# Seting the dataset path
# =========================================================================
DS_PATH = '/home/joao/Datasets/38-Cloud/38-Cloud_test'

# Sempre Teste
fold = 'test' # ['val', 'test']

save_image = True

# Experiment path
exp_path = os.path.join(args.main_path, args.exp)
print(exp_path)

# Caminho para o resultado (imagens)
pred_path = os.path.join(exp_path, f'{fold}_pred')
print(pred_path)

# Caminho para o GROUND-TRUTH (imagens)
gt_path = os.path.join(DS_PATH, 'Entire_scene_gts')

def pixel_accuracy_(output, mask):
    """ NumPy. 
    """
    ### correct = torch.eq(output, mask).int()
    correct = np.equal(output, mask).astype(int)
    ### accuracy = float(correct.sum()) / float(correct.numel())
    accuracy = float(correct.sum()) / float(correct.size)
        
    return accuracy

def mIoU_(pred_mask, mask, smooth=1e-10, n_classes=23):
        """ Versão NumPy. Funciona.
        """
        ### pred_mask = pred_mask.contiguous().view(-1)
        ### mask = mask.contiguous().view(-1)
        pred_mask = np.copy(pred_mask).reshape(-1)
        mask = np.copy(mask).reshape(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.astype(np.int64).sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = np.logical_and(true_class, true_label).sum().astype(float).item()
                union = np.logical_or(true_class, true_label).sum().astype(float).item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
                
        return np.nanmean(iou_per_class)


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    """ Versão PyTorch. Funciona.
    """
    ### print(pred_mask.dtype, pred_mask.min(), pred_mask.max())
    ### print(mask.dtype, mask.min(), mask.max())

    with torch.no_grad():
        ### pred_mask = F.softmax(pred_mask, dim=1)
        ### pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
                
        return np.nanmean(iou_per_class)

smp_metric_name_lists = [
        'F1 score',
        'IoU score or Jaccard index',
        'Accuracy',
        'Precision or positive predictive value (PPV)',
        'Sensitivity, recall, or true positive rate (TPR)',
        'Sensitivity, recall, or true positive rate (TPR)',
        'Specificity, or true negative rate (TNR)',
        'Balanced accuracy',
        'Precision or positive predictive value (PPV)',
        'Negative predictive value (NPV)',
        'Miss rate or false negative rate (FNR)',
        'False positive rate (FPR)',
        'False discovery rate (FDR)',
        'False omission rate (FOR)',
        'Positive likelihood ratio (LR+)',
        'Negative likelihood ratio (LR-)'
    ]

smp_metric_list = [
        smp.metrics.f1_score, # F1 score
        smp.metrics.iou_score, # IoU score or Jaccard index
        smp.metrics.accuracy, # Accuracy
        smp.metrics.precision, # Precision or positive predictive value (PPV)
        smp.metrics.recall, # Sensitivity, recall, hit rate, or true positive rate (TPR)
        smp.metrics.sensitivity, # Sensitivity, recall, hit rate, or true positive rate (TPR)
        smp.metrics.specificity, # Specificity, selectivity or true negative rate (TNR)
        smp.metrics.balanced_accuracy, # Balanced accuracy
        smp.metrics.positive_predictive_value, # Precision or positive predictive value (PPV)
        smp.metrics.negative_predictive_value, # Negative predictive value (NPV)
        smp.metrics.false_negative_rate, # Miss rate or false negative rate (FNR)
        smp.metrics.false_positive_rate, # Fall-out or false positive rate (FPR)
        smp.metrics.false_discovery_rate, # False discovery rate (FDR)
        smp.metrics.false_omission_rate, # False omission rate (FOR)
        smp.metrics.positive_likelihood_ratio, # Positive likelihood ratio (LR+)
        smp.metrics.negative_likelihood_ratio, # Negative likelihood ratio (LR-)
    ]

def compute_metrics_smp(smp_reduction):
    """
    """
    print('\n>> compute_metrics_smp')
    # Caminho para o arquivo CSV com os resultados
    results_path = os.path.join(exp_path, f'segmentation_details_{fold}_{smp_reduction}.csv')
    print(results_path)
    file_details = open(results_path, 'w')

    ### file_details.write(f'\nTest set')
    file_details.write('\nId;File path;pa;m_iu;pa*;ma*;m_iu*;fe_iu*')

    for metric_name in smp_metric_name_lists:
        file_details.write(f'; {metric_name}')

    filename_list = os.listdir(pred_path)

    # Para gerar um resultado consolidado.
    metric_list_list = []
    smp_metric_list_list = []

    for i, filename in enumerate(filename_list):
        print(f'Processing image {i}: {filename}')

        # Full predicted image
        pred_path_ = os.path.join(pred_path, filename)
        # Full ground-truth image
        gt_path_ = os.path.join(gt_path, 'edited_corrected_gts_' + filename)
        ### print(gt_path_)

        img_pred = plt.imread(pred_path_)
        img_pred = img_pred[:,:,0]

        img_gt = plt.imread(gt_path_)
        img_gt = img_gt[:,:,0]

        # PyTorch
        img_pred_ = torch.from_numpy(img_pred/255.).float()
        # NumPy
        ### img_pred_ = (img_pred/255.).astype(float)

        # PyTorch
        img_gt_ = torch.from_numpy(img_gt/255.).float()
        # NumPy
        ### img_gt_ = (img_gt/255.).astype(float)

        # Computa as métricas de validação.
        pa_ = pixel_accuracy_(img_pred, img_gt)
        # PyTorch
        m_iu_ = mIoU(img_pred_, img_gt_, n_classes=2)
        # NumPy
        ### m_iu_ = mIoU_(img_pred_, img_gt_, n_classes=2)
        
        # Computa as demais métricas de validação.
        pa = pixel_accuracy(img_pred, img_gt)
        ma = mean_accuracy(img_pred, img_gt)
        m_iu = mean_IU(img_pred, img_gt)
        fe_iu = frequency_weighted_IU(img_pred, img_gt)

        file_details.write(f'\n{i};{pred_path_};{pa_};{m_iu_};{pa};{ma};{m_iu};{fe_iu}')

        metric_list_list.append([pa_, m_iu_, pa, ma, m_iu, fe_iu])

        # SMP metrics
        # -----------
        tp, fp, fn, tn = smp.metrics.get_stats(img_pred_.int(), img_gt_.int(), mode='binary', threshold=0.5)

        smp_metric_list_ = []
        for j, smp_metric in enumerate(smp_metric_list):
            metric_ = smp_metric(tp, fp, fn, tn, smp_reduction)
            smp_metric_list_.append(metric_.cpu().numpy())
        
        smp_metric_list_list.append(smp_metric_list_)

        for smp_metric in smp_metric_list_:
            file_details.write(f';{smp_metric}')

    print('\nComputing mean and std for segmentation details...')
    ### print(metric_list_list)
    ### print(smp_metric_list_list)

    ### print(np.mean(metric_list_list, axis=0))
    ### print(np.mean(smp_metric_list_list, axis=0))

    # Compute mean 
    metric_list_mean = np.mean(metric_list_list, axis=0)
    smp_metric_list_mean = np.mean(smp_metric_list_list, axis=0)
    # Compute std
    ### metric_list_std = np.std(metric_list_list, axis=0)
    ### smp_metric_list_std = np.std(smp_metric_list_list, axis=0)
    # TEMP - Test
    metric_list_std = np.zeros(len(metric_list_list[0]))
    smp_metric_list_std = np.zeros(len(smp_metric_list_list[0]))


    # Save mean and std in the end of the file
    # ------------------------------------------------
    file_details.write(f'\n;Mean:')
    for metric_ in metric_list_mean:
        file_details.write(f';{metric_}')
    for smp_metric_ in smp_metric_list_mean:
        file_details.write(f';{smp_metric_}')

    file_details.write(f'\n;Std:')
    for metric_ in metric_list_std:
        file_details.write(f';{metric_}')
    for smp_metric_ in smp_metric_list_std:
        file_details.write(f';{smp_metric_}')

    file_details.close()

    return metric_list_mean, smp_metric_list_mean, metric_list_std, smp_metric_list_std

# Compute metrics
metric_list_mean_micro_iw, smp_metric_list_mean_micro_iw, \
metric_list_std_micro_iw, smp_metric_list_std_micro_iw = compute_metrics_smp('micro-imagewise')
metric_list_mean_macro_iw, smp_metric_list_mean_macro_iw, \
metric_list_std_macro_iw, smp_metric_list_std_macro_iw = compute_metrics_smp('macro-imagewise')
metric_list_mean_micro, smp_metric_list_mean_micro, \
metric_list_std_micro, smp_metric_list_std_micro = compute_metrics_smp('micro')
metric_list_mean_macro, smp_metric_list_mean_macro, \
metric_list_std_macro, smp_metric_list_std_macro = compute_metrics_smp('macro')
        
# Save images
# -----------
if save_image:
    print('\n>> Saving images...')

    # Path to the folder with the segmented images
    filename_list = os.listdir(pred_path)

    for i, filename in enumerate(filename_list):
        print(f'Processing image {i}: {filename}')

        # Cria as pastas para armazenar os resultados processados
        folder_names_list = [f'_({fold})_gt_over_image', f'_({fold})_pred_over_image', f'_({fold})_seg_map']

        for folder_ in folder_names_list:
            if not os.path.exists(os.path.join(exp_path, folder_)):
                os.makedirs(os.path.join(exp_path, folder_))

        # Full predicted image
        pred_path_ = os.path.join(pred_path, filename)
        # Full ground-truth image
        gt_path_ = os.path.join(gt_path, 'edited_corrected_gts_' + filename)
        ### print(gt_path_)

        img_pred = plt.imread(pred_path_)
        img_pred = img_pred[:,:,0]

        img_gt = plt.imread(gt_path_)
        img_gt = img_gt[:,:,0]

        # Computa as demais métricas de validação.
        pa = pixel_accuracy(img_pred, img_gt)
        ma = mean_accuracy(img_pred, img_gt)

        # Imagem original (Gray)
        # ----------------------
        img_rgb = io.imread(os.path.join(DS_PATH, 'Natural_False_Color', filename))
        img_gray = color.rgb2gray(img_rgb)

        # print(img_pred.shape, img_pred.dtype, img_pred.min(), img_pred.max())

        # Converte imagens para binária [0, 1]
        img_pred = (img_pred > 127).astype(int)
        img_gt = (img_gt > 127).astype(int)

        # Imagem de VALIDAÇÃO sobre imagem de GROUND-TRUTH
        # ------------------------------------------------
        img_pred_over_gt = color.label2rgb(img_pred, image=img_gt, bg_label=0)

        # Segmementation map (TP, TN, FP, FN)
        # ------------------------------------------------
        img_gt_ = img_gt * 2

        # TP (3 - verde), TN (0 - black), FP (2 - laranja), FN (1 - red)
        img_count = img_pred + img_gt_

        img_count_r = np.zeros([img_count.shape[0], img_count.shape[1]], dtype=np.uint8)
        img_count_g = np.zeros([img_count.shape[0], img_count.shape[1]], dtype=np.uint8)
        img_count_b = np.zeros([img_count.shape[0], img_count.shape[1]], dtype=np.uint8)

        img_count_r[img_count == 1] = 255
        img_count_g[img_count == 3] = 255

        img_count_r[img_count == 2] = 255
        img_count_g[img_count == 2] = 127

        img_count_rgb = np.zeros([img_count.shape[0], img_count.shape[1], 3], dtype=np.uint8)
        img_count_rgb[:,:,0] = img_count_r
        img_count_rgb[:,:,1] = img_count_g
        img_count_rgb[:,:,2] = img_count_b

        # gt_over_image
        # -------------
        img_gt_over_image_ = color.label2rgb(label=img_gt, image=img_gray, bg_label=0)
        img_gt_over_image = segmentation.mark_boundaries(img_gt_over_image_, img_gt)

        img_gt_over_image = util.img_as_ubyte(img_gt_over_image)

        # pred_over_image
        # ---------------
        img_pred_over_image_ = color.label2rgb(label=img_pred, image=img_gray, bg_label=0)
        img_pred_over_image = segmentation.mark_boundaries(img_pred_over_image_, img_pred)
        
        img_pred_over_image = util.img_as_ubyte(img_pred_over_image)

        # Salvando as imagens
        # =========================================================================
        # gt_over_image
        io.imsave(os.path.join(exp_path, f'_({fold})_gt_over_image', f'{pa:.4f}_{filename.split(".")[0]}.png'), img_gt_over_image)
        # pred_over_image
        io.imsave(os.path.join(exp_path, f'_({fold})_pred_over_image', f'{pa:.4f}_{filename.split(".")[0]}.png'), img_pred_over_image)
        # seg_map
        io.imsave(os.path.join(exp_path, f'_({fold})_seg_map', f'{pa:.4f}_{filename.split(".")[0]}.png'), img_count_rgb)


# General report
# --------------
def gen_general_rep(smp_reduction, metric_list_, smp_metric_list_, str_):
    """ Generating general report.
    """
    print('\n>> gen_general_rep') 
    # Caminho para o arquivo CSV com os resultados
    gen_rep_path = os.path.join(args.main_path, f'general_report_{fold}_smp_{smp_reduction}_{str_}.csv')
    print(gen_rep_path)

    if args.ec == 0:
        gen_rep_file = open(gen_rep_path, 'w')
        gen_rep_file.write('Id;File path;pa;m_iu;pa*;ma*;m_iu*;fe_iu*')
        # Header for SMP metrics
        for metric_name in smp_metric_name_lists:
            gen_rep_file.write(f';{metric_name}')
    else:
        gen_rep_file = open(gen_rep_path, 'a')

    gen_rep_file.write(f'\n{args.ec};{args.exp} ')
    # Standard metrics
    for metric_ in metric_list_:
        gen_rep_file.write(f';{metric_}')
    # SMP metrics
    for smp_metric_ in smp_metric_list_:
        gen_rep_file.write(f';{smp_metric_}')

gen_general_rep('micro-imagewise', metric_list_mean_micro_iw, smp_metric_list_mean_micro_iw, 'mean')
gen_general_rep('macro-imagewise', metric_list_mean_macro_iw, smp_metric_list_mean_macro_iw, 'mean')
gen_general_rep('micro', metric_list_mean_micro, smp_metric_list_mean_micro, 'mean')
gen_general_rep('macro', metric_list_mean_macro, smp_metric_list_mean_macro, 'mean')

gen_general_rep('micro-imagewise', metric_list_std_micro_iw, smp_metric_list_std_micro_iw, 'std')
gen_general_rep('macro-imagewise', metric_list_std_macro_iw, smp_metric_list_std_macro_iw, 'std')
gen_general_rep('micro', metric_list_std_micro, smp_metric_list_std_micro, 'std')
gen_general_rep('macro', metric_list_std_macro, smp_metric_list_std_macro, 'std')

print('\nDone! (gen-results-test)')
