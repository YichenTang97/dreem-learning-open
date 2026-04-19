import copy
import hashlib
import json
import os
import random as rd

from dreem_learning_open.logger.logger import log_experiment
from dreem_learning_open.utils.memmap_eeg import filter_memmap_signals_eeg_only
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
from dreem_learning_open.utils.indexed_run_complete import check_indexed_run_complete
from dreem_learning_open.utils.train_test_val_split import train_test_val_split
import shutil


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


def memmap_directory_ready(memmaps_dir: str) -> bool:
    """True when ``h5_to_memmaps`` finished for this hash (same check as ``run_cnn_rnn``)."""
    return (
        os.path.isfile(os.path.join(memmaps_dir, "groups_description.json"))
        and os.path.isfile(os.path.join(memmaps_dir, "features_description.json"))
    )


def _recover_test_record_id(description):
    if not isinstance(description, dict):
        return None
    records_split = description.get('records_split')
    if not isinstance(records_split, dict):
        records_split = {}
    test_records = records_split.get('test_records', [])
    if isinstance(test_records, list) and len(test_records) == 1 and isinstance(test_records[0], str):
        candidate = test_records[0]
        if candidate and os.sep not in candidate and '/' not in candidate and '\\' not in candidate:
            return candidate

    dataset_parameters = description.get('dataset_parameters')
    if not isinstance(dataset_parameters, dict):
        dataset_parameters = {}
    split = dataset_parameters.get('split')
    if not isinstance(split, dict):
        split = {}
    split_test = split.get('test')
    if isinstance(split_test, list) and len(split_test) == 1 and isinstance(split_test[0], str):
        return os.path.basename(os.path.normpath(split_test[0]))
    return None


def _is_run_complete(run_dir, description):
    ok, _ = check_indexed_run_complete(run_dir, description)
    return ok


def _find_incomplete_run_ids_by_test_record(save_folder):
    """
    Return map: test_record_id -> run_uuid for runs that are present but incomplete.
    If multiple incomplete runs exist for one test record, keep the most recent.
    """
    if not os.path.isdir(save_folder):
        return {}

    chosen = {}
    for run_uuid in os.listdir(save_folder):
        run_dir = os.path.join(save_folder, run_uuid)
        if not os.path.isdir(run_dir):
            continue

        description_path = os.path.join(run_dir, 'description.json')
        if not os.path.isfile(description_path):
            continue
        try:
            with open(description_path, 'r') as f:
                description = json.load(f)
        except Exception:
            continue
        if not isinstance(description, dict):
            continue

        test_record_id = _recover_test_record_id(description)
        if not test_record_id:
            continue
        if _is_run_complete(run_dir, description):
            continue

        metadata = description.get('metadata', {})
        last_activity = metadata.get('end') or metadata.get('begin') or int(os.path.getmtime(description_path))
        previous = chosen.get(test_record_id)
        if previous is None or last_activity > previous['last_activity']:
            chosen[test_record_id] = {'run_uuid': run_uuid, 'last_activity': last_activity}

    return {k: v['run_uuid'] for k, v in chosen.items()}


def run_experiments(experiments, experiments_directory, output_directory, datasets,
                    fold_to_run=None, force=True, error_tolerant=False,
                    skip_memmap_build=False, reuse_incomplete_uuids=False,
                    eeg_only=False, memmap_only=False):
    for experiment in experiments:
        experiment_directory = os.path.join(experiments_directory, experiment)
        memmaps_description = json.load(open(os.path.join(experiment_directory, 'memmaps.json')))
        for raw_memmap in memmaps_description:
            dataset = raw_memmap.get('dataset')
            if dataset not in datasets:
                continue
            memmap_description = copy.deepcopy(raw_memmap)
            del memmap_description['dataset']
            if eeg_only:
                memmap_description = filter_memmap_signals_eeg_only(memmap_description)
            base_name = memmap_description.get('name', experiment)
            exp_name = base_name + ('_eeg' if eeg_only else '')
            dataset_parameters = json.load(
                open(os.path.join(experiment_directory, 'dataset.json')))
            for dataset_parameter in dataset_parameters:
                if 'name' in dataset_parameter:
                    exp_name_bis = os.path.join(exp_name, dataset_parameter['name'])
                else:
                    exp_name_bis = exp_name
                dataset_setting = datasets[dataset]
                save_folder = os.path.join(output_directory, dataset, exp_name_bis)
                if os.path.exists(save_folder) and force and not memmap_only:
                    shutil.rmtree(save_folder)

                incomplete_run_ids_by_test_record = {}
                if reuse_incomplete_uuids and not force and not memmap_only:
                    incomplete_run_ids_by_test_record = _find_incomplete_run_ids_by_test_record(save_folder)

                description_hash = memmap_hash(memmap_description)
                dataset_dir = os.path.join(
                    dataset_setting['memmap_directory'],
                    description_hash
                )
                if not skip_memmap_build:
                    h5_to_memmaps(
                        records=[os.path.join(dataset_setting['h5_directory'], record) for record in
                                 os.listdir(dataset_setting['h5_directory'])],
                        memmap_description=memmap_description,
                        memmap_directory=dataset_setting['memmap_directory'],
                        parallel=False,
                        error_tolerant=error_tolerant)
                elif not os.path.isdir(dataset_dir):
                    raise FileNotFoundError(
                        'skip_memmap_build=True but memmap directory is missing: {!r}'.format(
                            dataset_dir))
                if not memmap_directory_ready(dataset_dir):
                    raise FileNotFoundError(
                        'Memmap directory is missing or incomplete: {!r}'.format(dataset_dir))

                if memmap_only:
                    n_records = len(
                        [n for n in os.listdir(dataset_dir) if '.json' not in n]
                    )
                    print(
                        'memmap-only: pipeline ready at {!r} ({} record folder(s)). '
                        'Skipping training.'.format(dataset_dir, n_records)
                    )
                    continue

                normalization = json.load(
                    open(os.path.join(experiment_directory, 'normalization.json')))
                trainer = json.load(open(os.path.join(experiment_directory, 'trainer.json')))
                transform = json.load(
                    open(os.path.join(experiment_directory, 'transform.json')))
                net = json.load(open(os.path.join(experiment_directory, 'net.json')))

                temporal_context = dataset_parameter['temporal_context']
                temporal_context_mode = dataset_parameter['temporal_context_mode']

                available_dreem_records = [
                    os.path.join(dataset_dir, record) for record in
                    os.listdir(dataset_dir) if '.json' not in record
                ]
                # build the folds
                rd.seed(2019)
                rd.shuffle(available_dreem_records)

                if dataset in ['dodo', 'mass_multi_channel', 'mass']:
                    if dataset == 'dodo':
                        N_FOLDS = 20
                    if dataset in ['mass_multi_channel', 'mass']:
                        N_FOLDS = 31
                    N_FOLDS = N_FOLDS - 1
                    FOLDS_SIZE = int(len(available_dreem_records) // N_FOLDS)
                    folds = [available_dreem_records[FOLDS_SIZE * x:FOLDS_SIZE * (x + 1)] for x
                             in
                             range(int(len(available_dreem_records) / FOLDS_SIZE + 1))]

                else:
                    # LOOV training
                    folds = [[record] for record in available_dreem_records]

                # Do not mutate `fold_to_run`: when it is None we must recompute per
                # (experiment, memmap, dataset_parameter) or later experiments reuse the
                # first experiment's fold count (wrong folds / skipped folds).
                if fold_to_run is None:
                    effective_fold_indices = [j for j, _ in enumerate(folds)]
                else:
                    effective_fold_indices = list(fold_to_run)

                for i, fold in enumerate(folds):
                    if i in effective_fold_indices:
                        other_records = [record for record in available_dreem_records if
                                         record not in fold]
                        rd.seed(2019 + i)
                        rd.shuffle(other_records)
                        train_records, val_records, _ = train_test_val_split(other_records,
                                                                             0.8, 0.2,
                                                                             0,
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
                                    'test': fold
                                },
                                'temporal_context': temporal_context,
                                'temporal_context_mode': temporal_context_mode,
                                'transform_parameters': transform

                            },
                            'save_folder': os.path.join(output_directory, dataset, exp_name_bis),
                        }

                        experiment_id = None
                        if reuse_incomplete_uuids and len(fold) == 1:
                            test_record_id = os.path.basename(os.path.normpath(fold[0]))
                            experiment_id = incomplete_run_ids_by_test_record.get(test_record_id)

                        log_experiment(**experiment_description, parralel=True,
                                       generate_memmaps=False,
                                       experiment_id=experiment_id)
