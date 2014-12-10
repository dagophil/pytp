import os
import h5py
import vigra
import numpy


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
    _tpd_train_feat = "train_feat"
    _tpd_test_raw_path = "test_raw_path"
    _tpd_test_raw_key = "test_raw_key"
    _tpd_test_gt_path = "test_gt_path"
    _tpd_test_gt_key = "test_gt_key"
    _tpd_test_feat = "test_feat"

    def __init__(self, file_name, use_abs_paths=False, check_file_exists=False, check_feat_dims=False):
        """
        Create a TrainPredictData instance by loading the given file.
        If the given file does not exist, it will be created.

        :param file_name: file name
        :param use_abs_paths: when adding new data sets, the absolute path will be used instead of the relative path
        :param check_file_exists: when adding new data sets, check if the added file exists
        :param check_feat_dims: when adding new features, check if the feature dimension matches the previous ones
        """
        # Check if the file can be opened with h5py.
        # If it does not exist, it will be created by h5py.
        f = h5py.File(file_name, "a")
        f.close()

        self.file_name = file_name
        self.use_abs_paths = use_abs_paths
        self.check_file_exists = check_file_exists
        self.check_feat_dims = check_feat_dims
        self.base_path = os.path.relpath(os.path.dirname(self.file_name))

    def _to_tpd_path(self, file_name):
        """
        Given a file name relative to the working directory, return the
        file path relative to the .tpd file considering the abs path setting.

        :param file_name: file name relative to working directory
        :return: file name relative to .tpd file considering abs path setting
        """
        if self.use_abs_paths:
            path = os.path.abspath(file_name)
        else:
            path = os.path.relpath(file_name, start=self.base_path)
        return path

    def _to_rel_path(self, file_name):
        """
        Given a file name relative to the .tpd file, return the
        file path relative to the current working directory.

        :param file_name: file name relative to the .tpd file or abs path
        :return: file name relative to the working directory
        """
        if os.path.isabs(file_name):
            path = os.path.relpath(file_name)
        else:
            path = os.path.join(self.base_path, file_name)
        return path

    def _check_file_exists(self, file_name):
        """
        If self.check_file_exists is False, this method does nothing.
        Otherwise, an exception will be raised if the given file does not exist.

        :param file_name: file name relative to the working directory
        """
        if self.check_file_exists:
            if not os.path.isfile(file_name):
                raise TPDError("_check_file_exists(): The given file does not exist.")

    def _set_data(self, file_name, h5_key, tpd_path, tpd_key):
        """Set file name and h5 key of the data.

        :param file_name: file name
        :param h5_key: h5 key
        :param tpd_path: tpd key of the file
        :param tpd_key: tpd key of the h5 key
        """
        self._check_file_exists(file_name)
        with h5py.File(self.file_name, "a") as f:
            if tpd_path in f.keys():
                del f[tpd_path]
            if tpd_key in f.keys():
                del f[tpd_key]
            f[tpd_path] = self._to_tpd_path(file_name)
            f[tpd_key] = h5_key

    def _get_data(self, tpd_path, tpd_key):
        """Return file name and h5 key of the data.

        :param tpd_path: tpd key of the file
        :param tpd_key: tpd key of the h5 key
        :return: file name, h5 key
        """
        with h5py.File(self.file_name, "r") as f:
            if tpd_path not in f.keys() or tpd_key not in f.keys():
                raise TPDError("_get_data(): The .tpd file does not contain the desired data.")
            file_name = f[tpd_path].value
            h5_key = f[tpd_key].value
        file_name = self._to_rel_path(file_name)
        return file_name, h5_key

    def set_train_raw(self, file_name, h5_key):
        """Set file name and h5 key of the training raw data.

        :param file_name: file name
        :param h5_key: h5 key
        """
        self._set_data(file_name, h5_key, self._tpd_train_raw_path, self._tpd_train_raw_key)

    def get_train_raw(self):
        """Return file name and h5 key of the training raw data.

        :return: file name, h5 key
        """
        return self._get_data(self._tpd_train_raw_path, self._tpd_train_raw_key)

    def get_train_raw_data(self):
        """Return the training raw data.

        :return: training raw data
        """
        file_name, h5_key = self.get_train_raw()
        return vigra.readHDF5(file_name, h5_key)

    def set_train_gt(self, file_name, h5_key):
        """Set file name and h5 key of the training gt data.

        :param file_name: file name
        :param h5_key: h5 key
        """
        self._set_data(file_name, h5_key, self._tpd_train_gt_path, self._tpd_train_gt_key)

    def get_train_gt(self):
        """Return file name and h5 key of the training raw data.

        :return: file name, h5 key
        """
        return self._get_data(self._tpd_train_gt_path, self._tpd_train_gt_key)

    def get_train_gt_data(self):
        """Return the training gt data.

        :return: training gt data
        """
        file_name, h5_key = self.get_train_gt()
        return vigra.readHDF5(file_name, h5_key)

    def set_test_raw(self, file_name, h5_key):
        """Set file name and h5 key of the test raw data.

        :param file_name: file name
        :param h5_key: h5 key
        """
        self._set_data(file_name, h5_key, self._tpd_test_raw_path, self._tpd_test_raw_key)

    def get_test_raw(self):
        """Return file name and h5 key of the test raw data.

        :return: file name, h5 key
        """
        return self._get_data(self._tpd_test_raw_path, self._tpd_test_raw_key)

    def get_test_raw_data(self):
        """Return the test raw data.

        :return: test raw data
        """
        file_name, h5_key = self.get_test_raw()
        return vigra.readHDF5(file_name, h5_key)

    def set_test_gt(self, file_name, h5_key):
        """Set file name and h5 key of the test gt data.

        :param file_name: file name
        :param h5_key: h5 key
        """
        self._set_data(file_name, h5_key, self._tpd_test_gt_path, self._tpd_test_gt_key)

    def get_test_gt(self):
        """Return file name and h5 key of the test gt data.

        :return: file name, h5 key
        """
        return self._get_data(self._tpd_test_gt_path, self._tpd_test_gt_key)

    def get_test_gt_data(self):
        """Return the test gt data.

        :return: test gt data
        """
        file_name, h5_key = self.get_test_gt()
        return vigra.readHDF5(file_name, h5_key)

    def _add_feature(self, file_name, h5_key, tpd_key):
        """
        Add the given feature to the training or test data, depending on the tpd key.
        The feature will only be added, if it is not already in the feature list.

        :param file_name: file name
        :param h5_key: h5 key
        :param tpd_key: key in the .tpd file
        """
        # TODO: Check feature dimension.
        self._check_file_exists(file_name)
        to_add = [self._to_tpd_path(file_name), h5_key]
        with h5py.File(self.file_name, "a") as f:
            if tpd_key in f.keys():
                feature_list = f[tpd_key].value.tolist()
                del f[tpd_key]
            else:
                feature_list = []
            if to_add not in feature_list:
                feature_list.append(to_add)
            f[tpd_key] = numpy.array(feature_list)

    def add_train_feature(self, file_name, h5_key):
        """Add the given training feature, but only if it is not already in the feature list.

        :param file_name: file name
        :param h5_key: h5 key
        """
        self._add_feature(file_name, h5_key, self._tpd_train_feat)

    def add_test_feature(self, file_name, h5_key):
        """Add the given test feature, but only if it is not already in the feature list.

        :param file_name: file name
        :param h5_key: h5 key
        """
        self._add_feature(file_name, h5_key, self._tpd_test_feat)

    def get_feature_data(self, tpd_key):
        """Get the features of the desired data set.

        :param tpd_key: tpd key of the data
        :return:
        """
        # TODO: Implement.
