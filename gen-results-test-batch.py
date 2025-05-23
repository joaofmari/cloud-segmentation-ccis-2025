import os
import time, datetime
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--main_path', type=str, default='exp_38cloud', help='Main path.')
### parser.add_argument('--fold', type=str, default='val', help='Select fold: [val, test].')

args = parser.parse_args()

exp_path_list = [path for path in os.listdir(args.main_path) if os.path.isdir(os.path.join(args.main_path, path))]
exp_path_list.sort()

ec = 0

### smp_reduction_list = ['micro-imagewise', 'macro-imagewise', 'micro', 'macro']

for exp_path in exp_path_list:
    print(exp_path[:13])

    # Only the folders generated by the Matlab script.
    if exp_path[:13] == 'entire_masks_':
        ### for smp_reduction in smp_reduction_list:
            ### cmd_str = f'nohup python gen-results-test.py --main_path {args.main_path} --exp {"entire_masks_" + exp_path}'
            ### cmd_str = f'nohup python gen-results-test.py --main_path {args.main_path} --exp {exp_path} --ec {ec} --smp_reduction {smp_reduction}'
            cmd_str = f'nohup python gen-results-test.py --main_path {args.main_path} --exp {exp_path} --ec {ec}'

            ec = ec + 1

            print(cmd_str)
            os.system(cmd_str)

if os.path.exists('./nohup.out'):
    suffix = ''
    while True:
        if os.path.exists(os.path.join(args.main_path, 'nohup' + suffix + '.out')):
            suffix += '_'
        else:
            break
    shutil.move('./nohup.out', os.path.join(args.main_path, 'nohup' + suffix + '.out'))

print('\nDone! (gen-results-batch)')


