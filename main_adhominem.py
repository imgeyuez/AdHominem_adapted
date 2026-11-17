# -*- coding: utf-8 -*-
from adhominem import AdHominem
import pickle
import os
from datetime import datetime, timedelta
import time
import random
import math



def main():

     # Hyperparameters
    hyper_parameters = {
        # Architecture
        'D_c': 10,                      # character embedding dimension
        'D_r': 30,                      # character representation dimension
        'w': 4,                         # length of 1D-CNN sliding window
        'D_w': 300,                     # dimension of word embeddings
        'D_s': 50,                      # dimension sentence embeddings
        'D_d': 50,                      # dimension of document embedding
        'D_mlp': 60,                    # final output dimension

        # Input dimensions
        'T_c': 15,                      # maximum number of characters per words
        'T_w': 20,                      # maximum number of words per sentence
        'T_s': 50,                      # maximum number of sentences per document

        # Similarity boundaries
        't_s': 0.95,                    # boundary for similar pairs (close to one)
        't_d': 0.05,                    # boundary for dissimilar pairs (close to zero)

        # Training
        'epochs': 50,                   # total number of epochs # default = 100
        'train_word_embeddings': False, # fine-tune pre-trained word embeddings
        'batch_size_tr': 32,            # batch size for training # default = 32
        'batch_size_dev': 32,
        'initial_learning_rate': 0.002, # initial learning rate

        # Dropout
        'keep_prob_cnn': 0.9,           # dropout for 1D-CNN layer
        'keep_prob_lstm': 0.9,          # variational dropout for BiLSTM layer
        'keep_prob_att': 0.9,           # dropout for attention layer
        'keep_prob_metric': 0.9,        # dropout for metric learning layer
    }


    # ---------------------------------------
    # Random Search Configuration
    # ---------------------------------------

    N_RUNS = 10
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)

    search_space = {
        # 'epochs': [20, 30, 50],
        'batch_size_tr': [16, 32, 64],
        # 'batch_size_dev': [32, 64, 128],

        'initial_learning_rate': (0.0001, 0.003),

        'keep_prob_cnn': [0.8, 0.9, 1.0],
        'keep_prob_lstm': [0.8, 0.9, 1.0],
        'keep_prob_att': [0.8, 0.9, 1.0],
        'keep_prob_metric': [0.8, 0.9, 1.0],
    }


    random_search_configs = list()

    for _ in range(N_RUNS):
        config = hyper_parameters.copy()
        for param, values in search_space.items():
            if param == 'initial_learning_rate':
                low, high = values

                config[param] = math.exp(random.uniform(math.log(low), math.log(high)))

            else:
                config[param] = random.choice(values)
                
        random_search_configs.append(config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Main directory for the entire random search
    dir_results = os.path.join('results', f'adhominem_random_search_{timestamp}')

    os.makedirs(dir_results, exist_ok=True)

    # # create folder for results
    # dir_results = os.path.join('results')
    # if not os.path.exists(dir_results):
    #     os.makedirs(dir_results)

    # checkpoint_dir = os.path.join(dir_results, f"adhominem_{timestamp}", "checkpoints")

    DATA_PATH = "/home/iyuzu/master_thesis/ad_hominem/AdHominem_adapted/data/VERANDA_files_2026-11-13_15-12-49.pkl"

    # load docs, vocabularies and initialized word embeddings
    with open(DATA_PATH, "rb") as f:
        (
            docs_L_tr, docs_R_tr, labels_tr, pair_ids_tr,
            docs_L_dev, docs_R_dev, labels_dev, pair_ids_dev,
            docs_L_te, docs_R_te, labels_te, pair_ids_te,
            V_w, E_w, V_c
        ) = pickle.load(f)

    # Prepare datasets (shared across all runs)
    train_set = (docs_L_tr, docs_R_tr, labels_tr)
    dev_set = (docs_L_dev, docs_R_dev, labels_dev, pair_ids_dev)

    # ---------------------------------------
    # Random Search: Train all configurations
    # ---------------------------------------

    search_start_datetime = datetime.now()
    search_start_time = time.perf_counter()

    for run_idx, config in enumerate(random_search_configs, start=1):
        print(f"\n{'=' * 60}")
        print(f"Starting Random Search Run {run_idx}/{N_RUNS}")
        print(f"{'=' * 60}")

        run_start_datetime = datetime.now()
        run_start_time = time.perf_counter()

        # Independent hyperparameters for this run
        run_hyper_parameters = config.copy()

        # Add vocabularies and dataset sizes
        run_hyper_parameters['V_w'] = V_w
        run_hyper_parameters['V_c'] = V_c
        run_hyper_parameters['N_tr'] = len(labels_tr)
        run_hyper_parameters['N_dev'] = len(labels_dev)

        # Separate output directory for each run
        run_dir = os.path.join(dir_results, f"run_{run_idx:02d}")

        os.makedirs(run_dir, exist_ok=True)

        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        file_results = os.path.join(run_dir, "results.txt")

        # Log hyperparameter configuration
        with open(file_results, 'w') as f:

            f.write(
                "--------------------------------------------------\n"
                "PARAMETER SETUP\n"
                "--------------------------------------------------\n"
            )

            for hp, value in run_hyper_parameters.items():

                if hp in ['V_c', 'V_w']:
                    f.write(f"num {hp}: {len(value)}\n")
                else:
                    f.write(f"{hp}: {value}\n")

        # Initialize a fresh model for this run
        adhominem = AdHominem(
            hyper_parameters=run_hyper_parameters,
            E_w_init=E_w
        )

        try:
            # Train and select best checkpoint using DEV
            adhominem.train_model(
                train_set,
                dev_set,
                file_results,
                checkpoint_dir
            )

        finally:
            # Always close TensorFlow session
            adhominem.sess.close()

        # Calculate run duration
        run_end_datetime = datetime.now()
        run_duration_seconds = time.perf_counter() - run_start_time

        run_duration_formatted = str(
            timedelta(seconds=round(run_duration_seconds))
        )

        # Log run duration
        run_summary = (
            f"\nRun {run_idx}/{N_RUNS} finished.\n"
            f"Start: {run_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"End: {run_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Duration: {run_duration_formatted}\n"
            f"Duration (seconds): {run_duration_seconds:.2f}\n"
        )

        print(run_summary)

    # ---------------------------------------
    # Random Search Summary
    # ---------------------------------------

    search_end_datetime = datetime.now()
    search_duration_seconds = time.perf_counter() - search_start_time

    search_duration_formatted = str(
        timedelta(seconds=round(search_duration_seconds))
    )

    search_summary = (
        f"\n{'=' * 60}\n"
        f"RANDOM SEARCH FINISHED\n"
        f"{'=' * 60}\n"
        f"Total runs: {N_RUNS}\n"
        f"Total duration: {search_duration_formatted}\n"
    )

    with open(os.path.join(dir_results, "random_search_summary.txt"), "w") as f:
        f.write(search_summary)


if __name__ == '__main__':
    main()
