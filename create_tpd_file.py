import sys
import os
from trainpredict import TrainPredictData
import h5py


def get_existing_file(msg, skip=False):
    """Shows msg and asks for input until the input is an existing file.

    :param msg: some message
    """
    inp = None
    while inp is None:
        inp = raw_input(msg)
        if skip and len(inp) == 0:
            return None
        if not os.path.isfile(inp):
            print "Not a file:", inp
            inp = None
    return inp


def get_nonexisting_file(msg, skip=False):
    """Shows msg and asks for input until the input is not an existing file.

    :param msg: some message
    """
    inp = None
    while inp is None:
        inp = raw_input(msg)
        if skip and len(inp) == 0:
            return None
        if os.path.isfile(inp):
            print "Is a file:", inp
            inp = None
    return inp


def extract_h5_key(file_name, msg):
    """
    Reads the given file using h5 py. If it contains only a single key, this
    key is returned. Otherwise the given msg is shown until a valid key is found.

    :param file_name: file name
    :param msg: some message
    :return: h5 key
    """
    with h5py.File(file_name, "r") as f:
        keys = f.keys()
    if len(keys) == 1:
        return keys[0]
    else:
        keys = [str(k) for k in keys]
        inp = None
        while inp is None:
            print "Choose one of the keys:", keys
            inp = raw_input(msg)
            if inp in keys:
                return inp
            else:
                inp = None


def main():
    """Ask the user for input to create a .tpd file.
    """
    tpd_file_name = get_nonexisting_file("Enter name of new tpd file: ")
    tpd = TrainPredictData(tpd_file_name)

    train_raw_path = get_existing_file("Enter training raw path: ", skip=True)
    if train_raw_path is not None:
        train_raw_key = extract_h5_key(train_raw_path, "Enter training raw h5 key: ")
        tpd.set_train_raw(train_raw_path, train_raw_key)

    train_gt_path = get_existing_file("Enter training gt path: ", skip=True)
    if train_gt_path is not None:
        train_gt_key = extract_h5_key(train_gt_path, "Enter training gt h5 key: ")
        tpd.set_train_gt(train_gt_path, train_gt_key)

    train_pred_path = get_existing_file("Enter training pred path: ", skip=True)
    if train_pred_path is not None:
        train_pred_key = extract_h5_key(train_pred_path, "Enter training pred h5 key: ")
        tpd.set_train_pred(train_pred_path, train_pred_key)

    train_feat_path = get_existing_file("Enter training feature path: ", skip=True)
    while train_feat_path is not None:
        train_feat_key = extract_h5_key(train_feat_path, "Enter training feature path: ")
        tpd.add_train_feature(train_feat_path, train_feat_key)
        train_feat_path = get_existing_file("Enter training feature path: ", skip=True)

    test_raw_path = get_existing_file("Enter test raw path: ", skip=True)
    if test_raw_path is not None:
        test_raw_key = extract_h5_key(test_raw_path, "Enter test raw h5 key: ")
        tpd.set_test_raw(test_raw_path, test_raw_key)

    test_gt_path = get_existing_file("Enter test gt path: ", skip=True)
    if test_gt_path is not None:
        test_gt_key = extract_h5_key(test_gt_path, "Enter test gt h5 key: ")
        tpd.set_test_gt(test_gt_path, test_gt_key)

    test_pred_path = get_existing_file("Enter test pred path: ", skip=True)
    if test_pred_path is not None:
        test_pred_key = extract_h5_key(test_pred_path, "Enter test pred h5 key: ")
        tpd.set_test_pred(test_pred_path, test_pred_key)

    test_feat_path = get_existing_file("Enter test feature path: ", skip=True)
    while test_feat_path is not None:
        test_feat_key = extract_h5_key(test_feat_path, "Enter test feature path: ")
        tpd.add_test_feature(test_feat_path, test_feat_key)
        test_feat_path = get_existing_file("Enter test feature path: ", skip=True)

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
