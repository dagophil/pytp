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
    _tpd_train_shape = "train_shape"
    _tpd_test_raw_path = "test_raw_path"
    _tpd_test_raw_key = "test_raw_key"
    _tpd_test_gt_path = "test_gt_path"
    _tpd_test_gt_key = "test_gt_key"
    _tpd_test_feat = "test_feat"
    _tpd_test_shape = "test_shape"

    @staticmethod
    def _check_file_exists(file_name):
        """Raise an exception if the given file does not exist.

        :param file_name: file name relative to the working directory
        """
        if not os.path.isfile(file_name):
            raise TPDError("_check_file_exists(): The given file does not exist.")

    @staticmethod
    def _get_count(shape):
        """Return the number of instances, given a shape.

        :param shape: shape
        :return: number of instances
        """
        from functools import reduce
        from operator import mul
        return reduce(mul, shape, 1)

    def __init__(self, file_name, use_abs_paths=False):
        """
        Create a TrainPredictData instance by loading the given file.
        If the given file does not exist, it will be created.

        :param file_name: file name
        :param use_abs_paths: when adding new data sets, the absolute path will be used instead of the relative path
        """
        # Check if the file can be opened with h5py.
        # If it does not exist, it will be created by h5py.
        f = h5py.File(file_name, "a")
        f.close()

        self.file_name = file_name
        self.use_abs_paths = use_abs_paths
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

    def _set_data(self, file_name, h5_key, tpd_path, tpd_key, tpd_shape):
        """Set file name and h5 key of the data.

        :param file_name: file name
        :param h5_key: h5 key
        :param tpd_path: tpd key of the file
        :param tpd_key: tpd key of the h5 key
        :param tpd_shape: tpd key with the shape of the data
        """
        self._check_file_exists(file_name)

        # Read the shape of the new data set.
        with h5py.File(file_name, "r") as f:
            if h5_key not in f.keys():
                raise TPDError("_set_data(): The given h5 key does not exist in the given file.")
            new_shape = f[h5_key].shape
        new_count = self._get_count(new_shape)

        with h5py.File(self.file_name, "a") as f:
            # Check if the data shapes match.
            if tpd_shape in f.keys():
                shape = f[tpd_shape]
                count = self._get_count(shape)
                if new_count != count:
                    raise TPDError("_set_data(): The numbers of instances do not match.")
            else:
                f[tpd_shape] = new_shape

            if tpd_path in f.keys():
                del f[tpd_path]
            f[tpd_path] = self._to_tpd_path(file_name)
            if tpd_key in f.keys():
                del f[tpd_key]
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
        self._set_data(file_name, h5_key,
                       self._tpd_train_raw_path, self._tpd_train_raw_key,
                       self._tpd_train_shape)

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
        self._set_data(file_name, h5_key,
                       self._tpd_train_gt_path, self._tpd_train_gt_key,
                       self._tpd_train_shape)

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
        self._set_data(file_name, h5_key,
                       self._tpd_test_raw_path, self._tpd_test_raw_key,
                       self._tpd_test_shape)

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
        self._set_data(file_name, h5_key,
                       self._tpd_test_gt_path, self._tpd_test_gt_key,
                       self._tpd_test_shape)

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

    def _add_feature(self, file_name, h5_key, tpd_key, tpd_shape):
        """
        Add the given feature to the training or test data, depending on the tpd key.
        The feature will only be added, if it is not already in the feature list.

        :param file_name: file name
        :param h5_key: h5 key
        :param tpd_key: key in the .tpd file
        :param tpd_shape: key in the .tpd file with the shape of the data
        """
        self._check_file_exists(file_name)
        to_add = [self._to_tpd_path(file_name), h5_key]

        # Read the data shape from the given file.
        with h5py.File(file_name, "r") as f:
            if h5_key not in f.keys():
                raise TPDError("_add_feature(): The given h5 key does not exist in the given file.")
            new_shape = f[h5_key].shape
        new_count = self._get_count(new_shape)

        # Append the new feature to the feature list.
        with h5py.File(self.file_name, "a") as f:
            # Get the current feature list.
            if tpd_key in f.keys():
                feature_list = f[tpd_key].value.tolist()
            else:
                feature_list = []

            # Get the shape of the old data.
            if tpd_shape in f.keys():
                shape = f[tpd_shape]
                count = self._get_count(shape)
                if new_count != count:
                    raise TPDError("_add_feature(): The numbers of instances do not match.")
            else:
                f[tpd_shape] = new_shape

            # Append the feature to the list.
            if to_add not in feature_list:
                feature_list.append(to_add)

            if tpd_key in f.keys():
                del f[tpd_key]
            f[tpd_key] = numpy.array(feature_list)

    def add_train_feature(self, file_name, h5_key):
        """Add the given training feature, but only if it is not already in the feature list.

        :param file_name: file name
        :param h5_key: h5 key
        """
        self._add_feature(file_name, h5_key, self._tpd_train_feat, self._tpd_train_shape)

    def add_test_feature(self, file_name, h5_key):
        """Add the given test feature, but only if it is not already in the feature list.

        :param file_name: file name
        :param h5_key: h5 key
        """
        self._add_feature(file_name, h5_key, self._tpd_test_feat, self._tpd_test_shape)

    def get_feature_data(self, tpd_key):
        """Get the features of the desired data set.

        :param tpd_key: tpd key of the data
        :return:
        """
        # TODO: Implement.
        raise NotImplementedError
