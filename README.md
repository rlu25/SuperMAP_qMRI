1. dataset: Training data / Testing data

2. get_data: Collect data from dataset
   Run RL_get_train.m for collecting training data
   Run RL_get_test.m for collecting testing data
   (need threshold for groundtruth map)

3. get_data_for_model: Preprocess data for deep learning model
   Run process_train.m for training data during training
   Run process_test.m for validation data during training
   Run process_all_test.m for testing data during testing

4. dl_model: Supermap_model
   Training:
       Run supermap_train.py
       Checkpoint will save to folder: results
   Testing:
       Run supermap_test.py, choose number of checkpoint
       result will save to folder: results (size of reuslt is 41x41)

5. save_result
   Run save_data.m
   Transfer crop size into fully size, save to corresponding name.
