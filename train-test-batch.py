import os
import time, datetime
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--ds', type=str, default='38cloud', help='Dataset name.')
parser.add_argument('--mode', type=str, default='train', help='Train or Eval. ["train", "eval"]')

args = parser.parse_args()

EXP_PATH_MAIN = f'exp_{args.ds}' # exp_38cloud_old-encoders

### mode = 'eval'  # 'train', 'eval'

bs = 24 # [16, 24]
lr = 0.0001
loss = 'crossentropy'
scheduler = 'plateau'

max_epochs = 1000 # 1000
ec = 0 # 0

model_list_ = ['Unet', 
               'PSPNet', 
               'Linknet',
               'MAnet',
              ]
backbone_list_ = [
                  'vgg16',                  # bs: 16
                  'resnet50',               # bs: 16
                  'inceptionv4',            # bs: 16
                  'densenet121',            # bs: 16
                  'mobilenet_v2',           # bs: 24
                  'efficientnet-b2',        # bs: 16
                 ]

### smp_reduction_list = ['micro-imagewise', 'macro-imagewise', 'micro', 'macro']

bs_dict = {'vgg16': 16, 'resnet50': 16, 'inceptionv4': 16, 'densenet121': 16, 'mobilenet_v2': 24, 'efficientnet-b2': 16}

# Inicia contagem de tempo deste época
time_start = time.time()

for model_ in model_list_:
    for backbone_ in backbone_list_:
        ### for smp_reduction in smp_reduction_list:
        cmd_str = f'nohup python train-test.py --dataset_name {args.ds} --mode {args.mode} ' + \
                  f'--model {model_} --backbone {backbone_} --loss {loss} --batch_size {bs_dict[backbone_]} ' + \
                  f'--lr {lr} --scheduler {scheduler} --no-hp_optim ' + \
                  f'--max_epochs {max_epochs} --ec {ec}'

        ec = ec + 1

        os.system(cmd_str)

time_exp = time.time() - time_start
time_exp_hms = str(datetime.timedelta(seconds = time_exp))
print(f'Time exp.: {time_exp} sec ({time_exp_hms})')

if os.path.exists('./nohup.out'):
    suffix = ''
    while True:
        if os.path.exists(os.path.join(EXP_PATH_MAIN, 'nohup' + suffix + '.out')):
            suffix += '_'
        else:
            break
    shutil.move('./nohup.out', os.path.join(EXP_PATH_MAIN, 'nohup' + suffix + '.out'))

print('\nDone! (train-test-batch)')


