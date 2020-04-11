from scipy import stats, linalg
from geosoup.common import Handler, Opt, np
import warnings


__all__ = ['Distance',
           'Mahalanobis',
           'Euclidean']


class Distance(object):
    """
    Parent class for all the distance type methods
    """

    def __init__(self,
                 samples=None,
                 names=None,
                 csv_file=None,
                 index=None):
        """
        Class constructor
        :param samples: List of sample dictionaries
        :param names: Name of the columns or dimensions
        :param csv_file: path of the csv file with sample data in columns
        :param index: Name of index column
        :return: _Distance object
        """
        self.samples = samples
        self.nsamp = len(samples)
        self.names = names
        self.csv_file = csv_file
        self.index = index

        if samples is not None:
            self.samples = samples
            self.nsamp = len(samples)

        elif csv_file is not None:
            self.samples = Handler(filename=csv_file).read_from_csv(return_dicts=True)
            self.nsamp = len(self.samples)
        else:
            warnings.warn('Empty Samples class initialized')
            self.samples = None
            self.nsamp = 0

        if self.nsamp > 0:
            self.index = list(range(self.nsamp))
        else:
            self.index = list()

        if names is not None:
            self.names = names
        elif self.samples is not None:
            self.names = list(self.samples[0])
        else:
            self.names = list()

        self.matrix = None
        self.center = None

    def __repr__(self):
        return "<Distance class object at {}>".format(hex(id(self)))

    def sample_matrix(self):
        """
        Method to convert sample dictionaries to sample matrix
        :return: Numpy matrix with columns as dimensions and rows as individual samples
        """
        # dimensions of the sample matrix
        nsamp = len(self.samples)
        nvar = len(self.names)

        if nsamp > 1:
            # copy data to matrix
            self.matrix = np.array([[Handler.string_to_type(self.samples[i][self.names[j]])
                                     for j in range(0, nvar)]
                                    for i in range(0, self.nsamp)])
        else:
            raise ValueError('Not enough samples to make a matrix object')

    def cluster_center(self,
                       method='median'):
        """
        Method to determine cluster center of the sample matrix
        :param method: Type of reducer to use. Options: 'mean', 'median', 'percentile_xx' where xx is 1-99
        :return: Cluster center (vector of column/dimension values)
        """
        if self.matrix is not None:
            if method == 'median':
                self.center = np.array(np.median(self.matrix, axis=0))[0]
            elif method == 'mean':
                self.center = np.array(np.mean(self.matrix, axis=0))[0]
            elif 'percentile' in method:
                perc = int(method.replace('percentile', '')[1:])
                self.center = np.array(np.percentile(self.matrix, perc, axis=0))[0]
            else:
                raise ValueError("Invalid or no reducer")
        else:
            raise ValueError("Sample matrix not found")


class Mahalanobis(Distance):
    """
    Class for calculating Mahalanobis distance from cluster center
    """

    def __init__(self,
                 samples=None,
                 names=None,
                 index=None):
        """
        Class constructor
        Class constructor
        :param samples: List of sample dictionaries
        :param names: Name of the columns or dimensions
        :param index: Name of index column
        :return: Mahalanobis object
        """

        super(Mahalanobis, self).__init__(samples,
                                          names,
                                          index)

        self.inverse = None
        self.distance = None

    def __repr__(self):
        return "<Mahalanobis class object at {}>".format(hex(id(self)))

    def covariance(self,
                   inverse=False):
        """
        Method to calculate a covariance matrix for a given sample matrix
        where rows are samples, columns are dimensions
        :param inverse: Should the inverse matrix be calculated
        :return: Covariance or inverse covariance matrix (numpy.matrix object)
        """
        cov_mat = np.cov(self.matrix,
                         rowvar=False)

        if inverse:
            # Inverse using SVD
            u, s, v = np.linalg.svd(cov_mat)

            try:
                return np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)

            except ValueError:
                return None
        else:
            return np.array(cov_mat)

    def difference(self,
                   transpose=False):
        """
        Method to calculate difference from scene center
        :return: matrix (numpy.ndarray)
        """
        center = self.center

        diff_matrix = np.apply_along_axis(lambda row: np.array(row) - center,
                                          axis=1,
                                          arr=np.array(self.matrix))

        if transpose:
            return diff_matrix.T
        else:
            return diff_matrix

    def calc_distance(self):
        """
        Method to calculate mahalanobis distance from scene center
        :return: scalar value
        """
        inv_cov_matrix = self.covariance(True)

        diff = self.difference(False)
        transpose_diff = self.difference(True)

        mdist = np.zeros(self.nsamp)

        for i in range(0, self.nsamp):

            if inv_cov_matrix is None:
                mdist[i] = np.nan
                continue

            prod = np.array(
                            np.dot(
                                np.dot(diff[i, :],
                                       inv_cov_matrix),
                                transpose_diff[:, i])
                        )[0][0]

            if prod < 0:
                mdist[i] = np.nan
            else:
                mdist[i] = np.sqrt(prod)

        return list(mdist)

    def partial_corr(self):
        """
        Partial Correlation using the linear regression approach to compute partial
        correlation among variables
        :returns Sample linear partial correlation coefficients between pairs of variables in self.matrix
        [i, j] contains the partial correlation of self.matrix[:, i] and self.matrix[:, j]
        """

        var_data = np.asarray(self.matrix)
        n_var = var_data.shape[1]
        part_corr = np.zeros((n_var, n_var), dtype=np.float)

        for var_indx in range(0, n_var):
            part_corr[var_indx, var_indx] = 1

            for comparison_var_indx in range(var_indx + 1, n_var):

                indices = np.ones(n_var, dtype=np.bool)
                indices[var_indx] = False
                indices[comparison_var_indx] = False

                beta_var = linalg.lstsq(var_data[:, indices], var_data[:, comparison_var_indx])[0]
                beta_comparison_var = linalg.lstsq(var_data[:, indices], var_data[:, var_indx])[0]

                result_var = var_data[:, comparison_var_indx] - var_data[:, indices].dot(beta_var)
                result_comparison_var = var_data[:, var_indx] - var_data[:, indices].dot(beta_comparison_var)

                correlation = stats.pearsonr(result_var, result_comparison_var)[0]
                part_corr[var_indx, comparison_var_indx] = part_corr[comparison_var_indx, var_indx] = correlation

        return part_corr


class Euclidean(Distance):
    """
    Class for calculating Euclidean distance
    """

    def __init__(self,
                 samples=None,
                 csv_file=None,
                 names=None):
        """
        :param csv_file: csv file that contains the samples
        :param samples: List of dictionaries
        """

        self.csv_file = csv_file

        self.index = None
        self.nfeat = None

        super(Euclidean, self).__init__(samples,
                                        names,
                                        csv_file,
                                        self.index)

        if self.samples is not None:
            self.sample_matrix()
        else:
            self.matrix = None

        self.grid = None

        self.distance_matrix = None

    def __repr__(self):
        return "<Euclidean class object at {} with {} samples>".format(hex(id(self)),
                                                                       str(len(self.samples)))

    @staticmethod
    def euc_dist(vec1,
                 vec2):
        """
        Method to calculate euclidean distance between two vectors
        :param vec1: first vector
        :param vec2: second vector
        :return: scalar
        """

        return np.linalg.norm(np.array(vec1) - np.array(vec2))

    @staticmethod
    def mat_dist(vec1, mat1):
        """
        Method to calculate euclidean distance between between a vector and all the vectors in a matrix
        :param vec1: vector
        :param mat1: matrix (numpy array of vectors)
        :return: numpy array of scalars
        """
        return np.apply_along_axis(lambda x: Euclidean.euc_dist(x, np.array(vec1)),
                                   1,
                                   mat1)

    def calc_dist_matrix(self,
                         approach=2,
                         verbose=False):
        """
        Method to calculate euclidean distance from each sample
         and make a matrix
        :return: 2d matrix
        """

        if verbose:
            Opt.cprint('Building distance matrix... ', newline='')

        if approach == 1:
            self.distance_matrix = np.apply_along_axis(lambda x: Euclidean.mat_dist(x, self.matrix),
                                                       1,
                                                       self.matrix)

        elif approach == 2:
            ndims = self.matrix.shape[1]

            temp_mat = np.zeros([self.matrix.shape[0], self.matrix.shape[0]], np.float32)

            for dim in range(ndims):
                arr = np.repeat(self.matrix[:, dim][:, np.newaxis], self.nsamp, 1)
                arr_ = arr.T
                temp_mat += (arr - arr_) ** 2

            self.distance_matrix = np.sqrt(temp_mat)

        else:
            raise ValueError('Unrecognized approach')

        if verbose:
            Opt.cprint('Done!')

    def proximity_filter(self,
                         thresh=None,
                         verbose=False):
        """
        method to remove points based on proximity threshold
        :param thresh: proximity threshold (default: 90th percentile) valid values: 1-99
        :param verbose: If steps should be displayed
        :return: None
        """
        if verbose:
            Opt.cprint('Applying proximity filter...')

        if thresh is None:
            thresh = self.cluster_center('percentile_90')
        elif 'percentile_' in thresh:
            thresh = self.cluster_center(thresh)
        elif thresh in (int, float):
            thresh = self.cluster_center('percentile_{}'.format(str(int(thresh))))
        else:
            if verbose:
                warnings.warn('Invalid thresh value.\n Using default: 90th percentile centroid vector.')
            thresh = self.cluster_center('percentile_90')

        # number of close proximities associated with each element
        n_proxim = np.apply_along_axis(lambda x: np.count_nonzero((x > 0.0) & (x < thresh)),
                                       0,
                                       self.distance_matrix)

        if verbose:
            Opt.cprint('Max group size : {} '.format(str(n_proxim.max())), newline='')
            Opt.cprint('Min group size : {} '.format(str(n_proxim.min())))

        # sort the indices in increasing order of n_proxim
        idx = []
        idx += np.argsort(n_proxim).tolist()
        idx_out = list()

        # find indices of elements that should be removed
        for ii in idx:
            if ii not in idx_out:
                arr = self.distance_matrix[ii, 0:(ii + 1)]
                temp_list = (np.where((arr < thresh) & (arr > 0.0))[0]).tolist()
                idx_out += temp_list
                idx_out = list(set(idx_out))

        # sort the indices in decreasing order for pop()
        pop_idx = sorted(list(set(idx_out)),
                         reverse=True)

        if verbose:
            Opt.cprint('Removing {} elements...'.format(str(len(pop_idx))))

        for pop_id in pop_idx:
            self.samples.pop(pop_id)

        self.nsamp = len(self.samples)
        self.index = list(range(self.nsamp))

    def apply_proximity_filter(self,
                               **kwargs):
        """
        Apply proximity filter at a given threshold
        :param kwargs:
                thresh: proximity threshold (default: 90th percentile) valid values: 1-99
                        default: 90
        :return: list of dictionaries
        """
        self.calc_dist_matrix()
        self.proximity_filter(**kwargs)

        return self.samples
