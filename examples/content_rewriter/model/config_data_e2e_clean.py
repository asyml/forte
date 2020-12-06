import os

from typing import Dict, Any

dataset_dir = 'model/e2e_data'  # 'e2ev14_demo'#'e2e_0512_max5'

modes = ['train', 'val', 'test']
mode_to_filemode = {
    'train': 'train',
    'val': 'valid',
    'test': 'test',
}
field_to_vocabname = {
    'x_value': 'x_value',
    'x_type': 'x_type',
    'x_associated': 'x_associated',
    'y_aux': 'y',
    'x_ref_value': 'x_value',
    'x_ref_type': 'x_type',
    'x_ref_associated': 'x_associated',
    'y_ref': 'y',
}
fields = list(field_to_vocabname.keys())
train_batch_size = 32
eval_batch_size = 32
batch_sizes = {
    'train': train_batch_size,
    'val': eval_batch_size,
    'test': 1,  # eval_batch_size,
}

datas: Dict[str, Any] = {}


# pylint: disable=global-statement
def set_datas():
    global datas
    datas = {
        mode: {
            'num_epochs': 1,
            'shuffle': mode == 'train',
            'batch_size': batch_sizes[mode],
            'allow_smaller_final_batch': mode != 'train',
            'datasets': [
                {
                    'files': [os.path.join(
                        dataset_dir, mode,
                        '{}.{}.txt'.format(field, mode_to_filemode[mode])
                    )],
                    'vocab_file': os.path.join(
                        dataset_dir,
                        '{}.vocab.txt'.format(field_to_vocabname[field])),
                    'data_name': field,
                }
                for field in fields]
        }
        for mode in modes
    }


set_datas()
