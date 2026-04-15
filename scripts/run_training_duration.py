import hashlib
import json
import os
import numpy as np

from dreem_learning_open.logger.logger import log_experiment
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
from dreem_learning_open.settings import DODH_SETTINGS
from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY
from dreem_learning_open.utils.train_test_val_split import train_test_val_split


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


datasets = {'dodh': DODH_SETTINGS}
experiment_name = "training_duration"
_script_dir = os.path.dirname(os.path.abspath(__file__))
experiments_directory = os.path.join(_script_dir, experiment_name)
models = [
    experiment for experiment in os.listdir(experiments_directory)
    if os.path.isdir(os.path.join(experiments_directory, experiment))
]

for model in models:
    experiment_directory = os.path.join(experiments_directory, model)
    memmaps_description = json.load(open(os.path.join(experiment_directory, 'memmaps.json')))
    for memmap_description in memmaps_description:
        dataset = memmap_description['dataset']
        del memmap_description['dataset']
        others = json.load(open(os.path.join(experiment_directory, 'dataset.json')))
        for other in others:
            dataset_setting = datasets[dataset]

            normalization = json.load(open(os.path.join(experiment_directory, 'normalization.json')))
            trainer = json.load(open(os.path.join(experiment_directory, 'trainer.json')))
            transform = json.load(open(os.path.join(experiment_directory, 'transform.json')))
            net = json.load(open(os.path.join(experiment_directory, 'net.json')))

            temporal_context, temporal_context_mode = other['temporal_context'], other[
                'temporal_context_mode']

            description_hash = memmap_hash(memmap_description)
            h5_dir = dataset_setting['h5_directory']
            h5_to_memmaps(
                records=[os.path.join(h5_dir, record) for record in os.listdir(h5_dir)],
                memmap_description=memmap_description,
                memmap_directory=dataset_setting['memmap_directory'],
                parallel=False)
            dataset_dir = os.path.join(dataset_setting['memmap_directory'], description_hash)
            available_dreem_records = [
                os.path.join(dataset_dir, record, '')
                for record in os.listdir(dataset_dir)
                if '.json' not in record
            ]

            train_records, val_records, test_record = train_test_val_split(available_dreem_records,
                                                                           train=0.70,
                                                                           test=0.25,
                                                                           val=0.05,
                                                                           seed=2019)

            experiment_description = {
                'memmap_description': memmap_description,
                'dataset_settings': dataset_setting,
                'trainer_parameters': trainer,
                'normalization_parameters': normalization,
                'net_parameters': net,
                'dataset_parameters': {
                    'split': {
                        'train': train_records,
                        'val': val_records,
                        'test': test_record,
                    },
                    'temporal_context': temporal_context,
                    'temporal_context_mode': temporal_context_mode,
                    'transform_parameters': transform

                },
                'save_folder': os.path.join(EXPERIMENTS_DIRECTORY, experiment_name, model)

            }
            save_folder = log_experiment(**experiment_description,
                                         parralel=True, generate_memmaps=False)

            training_folder = os.path.join(save_folder, "training")
            training_durations = [
                json.load(open(os.path.join(training_folder, metrics_path), "r"))["training_duration"]
                for metrics_path in os.listdir(training_folder)
                if "metrics" in metrics_path
            ]

            print(model, np.mean(training_durations), np.std(training_durations))
