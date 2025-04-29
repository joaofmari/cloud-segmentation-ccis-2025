# Evaluating multiple combinations of models and encoders to segment clouds in satellite images

* Leandro Henrique Furtado Pinto Silva - leandro.furtado@ufv.br 
* Mauricio Cunha Escarpinati - escarpinati@gmail.com
* André Ricardo Backes - arbackes@yahoo.com.br
* João Fernando Mari - joaof.mari@ufv.br
---
* Code for the paper published at CCIS 2024.
    * It is a extension of the paper published at VISAPP 2024 - Rome, Italy.
---

* Segmentation Models PyTorch
    * https://github.com/qubvel/segmentation_models.pytorch


## Configurando o ambiente de desenvolvimento
```
    $ conda update -n base -c defaults conda

    $ conda create -n env-smp-py39 python=3.9
    $ conda activate env-smp-py39

    $ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

    $ pip install segmentation-models-pytorch

    $ pip install jupyter
    $ pip install matplotlib
    $ pip install scikit-learn
    $ pip install scikit-image

    $ pip install opencv-python
    $ pip install -U albumentations --no-binary qudida,albumentations
    $ pip install torchsummary
```

### Information about the environment

```
    $ conda list --explicit > env-smp-py39.txt
```

### Saving environment

```
    $ conda env export > env-smp-py39.yml
```

### Loading the environment from the provided .yml file

```
    $ conda env create -f env-smp-py39.yml
```

### if no GPU available


## Datasets

WARNING: Em algumas distribuições do 38-Cloud existe um arquivo com extensão .tif (em mínuscula) em meio a extensões .TIF (maiusculas). 
Em sistemas case-sensitive podem ocorrer problemas.
SOLUÇÃO: Renomear o arquivo.

# Pre-trained backbones
* We provide the pre-trainned backbones in the file "checkpoints.zip" because some repositories are unavailable.
* Copy these files to ~/.cache/torch/hub/checkpoints if needed.

## Instruções

1. Para desenvolvimento e prototipação utilize os arquivos .ipynb (train_test.ipynb)
1. Para executar os experimentos, converter o arquivo .ipynb para .py.
1. Após a conversão, comentar a linha: sys.argv = ['-f']
    * A linha está identifica com o seguinte comentário:
```
    # ***** IMPORTANTE!!! *****
    # Comentar esta linha após gerar o arquivo .py!
    # *************************
    sys.argv = ['-f']
```

4. Para executar um único experimento use o arquivo train-test.py. 
    * Use o nohup quando rodar em servidor remoto (ssh):
    * Passe os argumentos adequados via linha de comando.
```
    $ nohup python train_test.py &
```
5. Para executar uma sequencia de experimentos modifique e execute o script 'train_test_batch.py'
6. Para executar apenas o teste de um modelo, passe o argumento '--mode eval'. 
   * Para isso, deve haver um arquivo com o modelo na pasta do exeperimento.
   * Cuidado, resultados anteriores serão sobrescritos.
7. Para consolidar os resultados sobre o conjunto de validação usar o 'gen-results-val.ipynb'.
    * Para executar em lote use o script 'gen-results-val-batch.py'.
    * Para isso é necessário converter o notebook 'gen-results-val.ipynb' para .py. Não esquecer de comentar a linha sys.argv = '['f']
    * Use a argumento de linha de comando fold como o valor val.
```
    $ nohup python gen-results-val-batch.py --fold val
```
8. Para gerar os resultados sobre o conjunto de testes execute o script 'evaluate_cloud.m' no Matlab. 
    * O script está configurado para executar no Windows.
    * Altere os caminhos para os arquivos nas variáveis:
        * gt_folder_path
        * preds_folder_root
    * É necessário executar o script para cada experimentos.
        * Descomente o caminho para o experimento desejado e execute o script.
            * Váriavel: preds_folder
        * Se necessário ajustar os caminhos para os experimentos.
9. Para consollidar os resultados do conjunto de testes executar o notebook 'gen-results-test.ipynb'.
    * Para executar em lote executar o script 'gen-results-batch.py'.
    * Para isso é necessário converter o notebook 'gen-results-test.ipynb' para .py. Não esquecer de comentar a linha sys.argv = '['f']
    * Use a argumento de linha de comando fold como o valor test.
```
    $ nohup python gen-results-batch.py --fold test
```


## Utilities:

* Memória da GPU não libera, mesmo nvidia-smi não mostrando nenhum processo:
    * https://stackoverflow.com/a/46597252

```
fuser -v /dev/nvidia*
                     USER        PID ACCESS COMMAND
/dev/nvidia0:        joao      44525 F...m python
/dev/nvidiactl:      joao      44525 F...m python
/dev/nvidia-uvm:     joao      44525 F...m python

kill -9 44525
```

* List Python processes
```
    $ ps -ef | grep python
```

## References

### Main reference:
* Tutorial:
    * Semantic Segmentation is Easy with Pytorch
        * https://www.kaggle.com/code/ligtfeather/semantic-segmentation-is-easy-with-pytorch
        * https://www.kaggle.com/code/ligtfeather/semantic-segmentation-is-easy-with-pytorch/notebook

* Dataset:
    * Aerial Semantic Segmentation Drone Dataset
        * https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset

### Other references:
1. https://medium.com/analytics-vidhya/how-to-create-a-custom-dataset-loader-in-pytorch-from-scratch-for-multi-band-satellite-images-c5924e908edf
1. https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
1. cars segmentation (camvid) [Segmentation Models PyTorch]
    * https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
1. How I built Supervised Skin Lesion Segmentation on HAM10000 Dataset
    * https://pub.towardsai.net/how-i-built-supervised-skin-lesion-segmentation-on-ham10000-dataset-f293c3239746


## Observações sobre os modelos

* Resultados (07/08/2023): Google Acadêmico: "semantic segmentation" \<arch> 

| Modelo     | Resultados | Obs
|------------| ---------- | --- |
| Unet       | 32.200     |
| Unet++     | 391        |
| MAnet      | 671        |
| Linknet    | 6.160      |
| FPN        | 8.180      | *
| PSPNet     | 8.070      |
| PAN        | 627        | *
| DeepLabV3  | 11.000     |
| DeepLabV3+ | 1.400      |

* (*) Search: "Pyramid Attention Network" AND "PAN"
* (*) Search: "Feature Pyramid Network" AND "FPN"

https://paperswithcode.com/paper/comprehensive-comparison-of-deep-learning/review/

## Pre-selected models

https://smp.readthedocs.io/en/latest/encoders.html

* For DeepLabV3, DeepLabV3+, and PAN dilation support is needed also
---