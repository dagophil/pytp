import os
import h5py
import vigra


class TPDError(RuntimeError):
    pass


class TrainPredictData(object):
    """
    The TrainPredictData class can be used to store and load .tpd files (data
    which is associated with a train / predict workflow).
    """

    # The paths that are used inside the .tpd file.
    _tpd_train_raw_path = "train_raw_path"
    _tpd_train_raw_key = "train_raw_key"
    _tpd_train_gt_path = "train_gt_path"
    _tpd_train_gt_key = "train_gt_key"
    _tpd_test_raw_path = "test_raw_path"
    _tpd_test_raw_key = "test_raw_key"
    _tpd_test_gt_path = "test_gt_path"
    _tpd_test_gt_key = "test_gt_key"

    def __init__(self, file_name, abs_paths=False):
        """
        Create a TrainPredictData instance by loading the given file.
        If the given file does not exist, it will be created.

        :param file_name: file name
        """
        # Check if the file can be opened with h5py.
        # If it does not exist, it will be created by h5py.
        f = h5py.File(file_name, "a")
        f.close()

        self.file_name = file_name
        self.abs_paths = abs_paths

    def set_train_raw(self, file_name, h5_key):
        """Set file name and h5 key of the training raw data.

        :param file_name: file name
        :param h5_key: h5 key
        """
        with h5py.File(self.file_name, "a") as f:
            f[self._tpd_train_raw_path] = file_name
            f[self._tpd_train_raw_key] = h5_key

    def get_train_raw(self):
        """Return file name and h5 key of the training raw data.

        :return: file name, h5 key
        """
        with h5py.File(self.file_name, "r") as f:
            if self._tpd_train_raw_path not in f.keys() or self._tpd_train_raw_key not in f.keys():
                raise TPDError("get_train_raw(): The .tpd file does not contain raw training data.")
            file_name = f[self._tpd_train_raw_path].value
            h5_key = f[self._tpd_train_raw_key].value
        return file_name, h5_key

    def get_train_raw_data(self):
        """Return the training raw data.

        :return: training raw data
        """
        file_name, h5_key = self.get_train_raw()
        return vigra.readHDF5(file_name, h5_key)

    def set_test_raw(self, file_name, h5_key):
        """Set file name and h5 key of the test raw data.

        :param file_name: file name
        :param h5_key: h5 key
        """
        with h5py.File(self.file_name, "a") as f:
            f[self._tpd_test_raw_path] = file_name
            f[self._tpd_test_raw_key] = h5_key

    def get_test_raw(self):
        """Return file name and h5 key of the test raw data.

        :return: file name, h5 key
        """
        with h5py.File(self.file_name, "r") as f:
            if self._tpd_test_raw_path not in f.keys() or self._tpd_test_raw_key not in f.keys():
                raise TPDError("get_test_raw(): The .tpd file does not contain raw test data.")
            file_name = f[self._tpd_test_raw_path].value
            h5_key = f[self._tpd_test_raw_key].value
        return file_name, h5_key

    def get_test_raw_data(self):
        """Return the test raw data.

        :return: test raw data
        """
        file_name, h5_key = self.get_test_raw()
        return vigra.readHDF5(file_name, h5_key)
