pytp
====

This repository contains the python TrainPredictData class.
The class can be used to store data which is associated with a train/predict workflow in a single h5 file.

Why pytp?
=========

When you do machine learning, you often have several data sets, some of them
used as training data, some as test data, some is ground truth data, etc.
Using these data sets in your python scripts might become confusing: If you
decide to use other training / test data, all of the file paths in the scripts
must be modified.

With pytp, all you have to do is create on .tpd file that stores some of your
data sets. If you decide to change training / test data for a single script,
all you have to do is to exchange the .tpd file. If you decide to change
training / test data for all of your scripts, you can modify the .tpd file.

Examples
========

Read training raw and ground truth data of an existing .tpd file:

    tpd = TrainPredictData(file_name)
    raw = tpd.get_train_raw_data()
    gt = tpd.get_train_gt_data()

Set training ground truth data of an existing .tpd file:

    tpd = TrainPredictData(file_name)
    tpd.set_train_gt(path, h5key)
   
Create a new .tpd file and add training raw data:

    tpd = TrainPredictData(file_name)
    tpd.set_train_raw(path, h5key)

Add some training features to an existing .tpd file:

    tpd = TrainPredictData(file_name)
    tpd.add_train_feature(path, h5key)
    tpd.add_train_feature(path2, h5key2)

Just create the .tpd file
=========================

If you only want to create a single .tpd file, you can use the script
create_tpd_file.py. You are then asked for the paths of the input files.
If those files only contain a single h5 key, this key is used. Else you have to
enter the correct key manually.

    python create_tpd_file.py

To skip a data set, just enter nothing.
