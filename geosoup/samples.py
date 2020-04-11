import warnings
from scipy.stats.stats import pearsonr
from geosoup.common import Handler, Opt, Sublist, Timer, np


__all__ = ['Samples']


class Samples:
    """
    Class to read and arrange sample data.
    Stores label and label names in y and y_names
    Stores feature and feature names in x and x_names.
    Currently the user has to provide sample csv files with one column as label (output)
    and the rest of the columns as feature attributes. There should be no index number column.
    All columns should be data only.
    """

    def __init__(self,
                 csv_file=None,
                 label_colname=None,
                 x=None,
                 y=None,
                 x_name=None,
                 y_name=None,
                 weights=None,
                 weights_colname=None,
                 use_band_dict=None,
                 max_allow_x=1e13,
                 max_allow_y=1e13,
                 **kwargs):

        """
        :param csv_file: csv file that contains the features (training or validation samples)
        :param label_colname: column in csv file that contains the feature label (output value)
        :param x: 2d array containing features (samples) without the label
        :param y: 1d array of feature labels (same order as x)
        :param x_name: 1d array of feature names (bands)
        :param y_name: name of label
        :param use_band_dict: list of attribute (band) names
        :param max_allow_x: Maximum allowed values of x
        :param max_allow_y: Maximum allowed value of y
        """
        self.csv_file = csv_file
        self.label_colname = label_colname

        if type(x).__name__ in ('ndarray', 'NoneType'):
            self.x = x
        else:
            self.x = np.array(list(x))

        self.x_name = x_name

        if type(y).__name__ in ('ndarray', 'NoneType'):
            self.y = y
        else:
            self.y = np.array(list(y))

        self.y_name = y_name

        self.weights = weights
        self.weights_colname = weights_colname
        self.use_band_dict = use_band_dict

        self.index = None
        self.nfeat = None

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

        self.max_allow_x = max_allow_x
        self.max_allow_y = max_allow_y

        # either of label name or csv file is provided without the other
        if (csv_file is None) and (label_colname is None):
            pass  # warnings.warn("Samples class initiated without data file or label")

        # label name or csv file are provided
        elif (label_colname is not None) and (csv_file is not None):

            temp = Handler(filename=csv_file).read_from_csv()

            # label name doesn't match
            if any(label_colname in s for s in temp['name']):
                loc = temp['name'].index(label_colname)
            else:
                raise ValueError("Label name mismatch.\nAvailable names: " + ', '.join(temp['name']))

            # read from data dictionary
            self.x_name = Sublist(elem.strip() for elem in temp['name'][:loc] + temp['name'][(loc + 1):])
            self.x = np.array(list(feat[:loc] + feat[(loc + 1):] for feat in temp['feature']))
            self.y = np.array(list(feat[loc] for feat in temp['feature']))
            self.y_name = temp['name'][loc].strip()

            # if band name dictionary is provided
            if use_band_dict is not None:
                self.y_name = [use_band_dict[b] for b in self.y_name]

        elif (label_colname is None) and (csv_file is not None):

            temp = Handler(filename=csv_file).read_from_csv()

            # read from data dictionary
            self.x_name = Sublist(temp['name'])
            self.x = np.array(list(feat for feat in temp['feature']))

        else:
            ValueError("No data found for label.")

        if self.x is not None and self.y is not None:
            if self.y_name is None:
                self.y_name = 'y'
            if (self.x_name is None) or \
                    (type(self.x_name) not in (list, tuple)) or \
                    (len(self.x_name) != self.x.shape[1]):
                self.x_name = list('x{}'.format(str(i+1)) for i in range(self.x.shape[1]))

        if weights is None:
            if weights_colname is not None:
                if csv_file is not None:

                    temp = Handler(filename=csv_file).read_from_csv()

                    # label name doesn't match
                    if any(weights_colname in n for n in temp['name']):
                        loc = temp['name'].index(weights_colname)
                    else:
                        raise ValueError("Weight column name mismatch.\nAvailable names: " + ', '.join(temp['name']))

                    self.weights = self.x[:, loc]
                    self.x = np.delete(self.x, loc, 1)

                else:
                    raise ValueError("No csv_file specified for weights")

        # if keywords are supplied
        if kwargs is not None:

            # columns containing data
            if 'columns' in kwargs:
                if type(kwargs['columns']).__name__ == 'list':
                    self.columns = np.array(kwargs['columns'])
                elif type(kwargs['columns']).__name__ in ('ndarray', 'NoneType'):
                    self.columns = kwargs['columns']
                else:
                    self.columns = np.array(list(kwargs['columns']))
            else:
                self.columns = None

            # IDs of samples
            if 'ids' in kwargs:
                self.ids = kwargs['ids']
            else:
                self.ids = None

        else:
            self.columns = None
            self.ids = None

        if self.x is not None:

            if self.columns is None:
                self.columns = np.arange(0, self.x.shape[1])

            self.nsamp = self.x.shape[0]
            self.nvar = self.x.shape[1]

            self.nfeat = self.x.shape[1]

            self.xmin = self.x.min(0, initial=max_allow_x)
            self.xmax = self.x.max(0, initial=max_allow_y)

            self.index = np.arange(0, self.x.shape[0])

        else:
            self.nsamp = 0
            self.nvar = 0

        if self.y is not None:
            self.ymin = self.y.min(initial=-max_allow_y)
            self.ymax = self.y.max(initial=max_allow_y)

        if self.y is not None:
            self.head = '\n'.join(list(str(elem) for elem in
                                       [' '.join(list(self.x_name) + [self.y_name])] +
                                       list(' '.join(list(str(elem_) for elem_ in self.x[i, :].tolist() + [self.y[i]]))
                                            for i in range(10))))
        else:
            self.head = '<empty>'

    def __repr__(self):
        """
        Representation of the Samples object
        :return: Samples class representation
        """
        if self.csv_file is not None:
            return "<Samples object from {cf} with {v} variables, {n} samples>".format(cf=Handler(
                                                                                       self.csv_file).basename,
                                                                                       n=len(self.x),
                                                                                       v=len(self.x_name))
        elif self.csv_file is None and self.x is not None:
            return "<Samples object with {n} samples>".format(n=len(self.x))
        else:
            return "<Samples object: __empty__>"

    def format_data(self):
        """
        Method to format the samples to the RF model fit method
        :param self
        :return: dictionary of features and labels
        """
        if self.columns is not None:
            column_list = []
            column_list += self.columns.tolist()
            out_x = self.x[:, self.columns]
            out_x_name = Sublist(self.x_name[i] for i in column_list)
        else:
            out_x = self.x
            out_x_name = self.x_name

        return {
            'features': out_x.copy(),
            'labels': self.y.copy(),
            'label_name': Opt.__copy__(self.y_name),
            'feature_names': Opt.__copy__(out_x_name),
        }

    def correlation_matrix(self,
                           display_progress=True):
        """
        Method to return a dictionary with correlation data
        rows = columns = variables (or dimensions)
        :param display_progress: Should the elements of correlation matrix
        be displayed while being calculated? (default: True)

        :return: Dictionary
        """
        # get data from samples
        data_mat = self.x
        nsamp, nvar = data_mat.shape
        print(nsamp, nvar)

        # get names of variables
        var_names = list()
        for i, name in enumerate(self.x_name):
            print(str(i)+' '+name)
            var_names.append(name.upper())

        # initialize correlation matrix
        corr = np.zeros([nvar, nvar], dtype=float)

        # calculate correlation matrix
        for i in range(0, nvar):
            for j in range(0, nvar):
                corr[i, j] = np.abs(pearsonr(Sublist.column(data_mat, i),
                                             Sublist.column(data_mat, j))[0])
                if display_progress:
                    str1 = '{row} <---> {col} = '.format(row=var_names[i], col=var_names[j])
                    str2 = '{:{w}.{p}f}'.format(corr[i, j], w=3, p=2)
                    print(str1 + str2)

        return {'data': corr, 'names': var_names}

    def merge_data(self,
                   samp):
        """
        Merge two sample sets together
        column and label names and orders should be the same in the two datasets
        :param self, samp:
        """
        self.x = np.vstack((self.x, samp.x))
        self.y = np.hstack((self.y, samp.y))
        self.nsamp = self.x.shape[0]
        self.index = np.arange(0, self.nsamp)
        self.xmin = self.x.min(0, initial=-self.max_allow_x)
        self.xmax = self.x.max(0, initial=self.max_allow_x)
        self.ymin = self.y.min(initial=-self.max_allow_y)
        self.ymax = self.y.max(initial=self.max_allow_y)

    @Timer.timing(True)
    def delete_column(self,
                      column_id=None,
                      column_name=None):
        """
        Function to remove a data column from the samples object
        :param column_id: ID (index) of the column
        :param column_name: Column label or name
        :return: Samples object with a column removed
        """
        if column_name is None and column_id is None:
            raise AttributeError('No argument for delete operation')

        elif column_id is None and column_name is not None:
            column_id = Sublist(self.x_name) == column_name

        self.x = np.delete(self.x, column_id, 1)

        self.x_name = self.x_name.remove(column_id)
        self.columns = np.arange(0, self.x.shape[1])
        self.nvar = self.x.shape[1]

    def extract_column(self,
                       column_id=None,
                       column_name=None):
        """
        Function to extract a data column from the samples object
        :param column_id: ID (index) of the column
        :param column_name: Column label or name
        :return: Samples object with only one column
        """

        if column_name is None and column_id is None:
            raise AttributeError('No argument for extract operation')

        elif column_id is None and column_name is not None:
            column_id = Sublist(self.x_name) == column_name

        return {'name': self.x_name[column_id], 'value': self.x[:, column_id]}

    def add_column(self,
                   column_name=None,
                   column_data=None,
                   column_order=None):
        """
        Function to add a column to the samples matrix.
        Column_order keyword is used after appending the column name and data to the right of the matrix
        but if column_data is None, self.x is re-ordered according to column_order
        :param column_name: Name of column to be added
        :param column_data: List of column values to be added
        :param column_order: List of numbers specifying column order for the column to be added
                            (e.g. if for three samples, the first value in column_data
                            is for second column, second value for first, third value for third,
                            the column_order is [1, 0, 2]
        :return: Samples object with added column
        """

        if column_data is None:
            if column_order is not None:
                self.x = self.x[:, np.array(column_order)]
                self.x_name = list(self.x_name[i] for i in column_order)
                return
            else:
                RuntimeError('No argument for add operation')
        else:
            column_data_ = np.array(column_data)
            self.x = np.hstack((self.x, column_data_[:, np.newaxis]))

            if column_name is None:
                column_name = 'Column_{}'.format(str(len(self.x_name) + 1))

            self.x_name.append(column_name)

            if column_order is None or len(column_order) != self.x.shape[1]:
                warnings.warn('Inconsistent or missing order - ignored')
                column_order = list(range(0, self.x.shape[1]))

            self.x = self.x[:, np.array(column_order)]
            self.x_name = list(self.x_name[i] for i in column_order)

            self.columns = list(range(0, self.x.shape[1]))
            self.nvar = len(self.columns)

    def save_to_file(self,
                     out_file):
        """
        Function to save sample object to csv file
        :param out_file: CSV file full path (string)
        :return: Write to file
        """

        out_arr = np.hstack((self.x, self.y[:, np.newaxis]))
        out_names = self.x_name + list(self.y_name)

        Handler(out_file).write_numpy_array_to_file(np_array=out_arr,
                                                    colnames=out_names)

    def random_partition(self,
                         percentage=75):

        """
        Method to randomly partition the samples based on a percentage
        :param percentage: Partition percentage (default: 75)
        (e.g. 75 for 75% training samples and 25% validation samples)
        :return: Tuple (Training sample object, validation sample object)
        """

        ntrn = int((percentage * self.nsamp) / 100.0)

        # randomly select training samples based on number
        trn_sites = np.random.choice(self.index,
                                     size=ntrn,
                                     replace=False)
        val_sites = self.index[~np.in1d(self.index, trn_sites)]

        # training sample object
        trn_samp = Samples()
        trn_samp.x_name = self.x_name
        trn_samp.y_name = self.y_name
        trn_samp.x = self.x[trn_sites, :]
        trn_samp.y = self.y[trn_sites]
        trn_samp.nsamp = trn_samp.x.shape[0]
        trn_samp.index = np.arange(0, trn_samp.nsamp)
        trn_samp.nfeat = trn_samp.x.shape[1]

        trn_samp.xmin = trn_samp.x.min(0, initial=-self.max_allow_x)
        trn_samp.xmax = trn_samp.x.max(0, initial=self.max_allow_x)
        trn_samp.ymin = trn_samp.y.min(initial=-self.max_allow_y)
        trn_samp.ymax = trn_samp.y.max(initial=self.max_allow_y)

        # validation sample object
        val_samp = Samples()
        val_samp.x_name = self.x_name
        val_samp.y_name = self.y_name
        val_samp.x = self.x[val_sites, :]
        val_samp.y = self.y[val_sites]
        val_samp.nsamp = val_samp.x.shape[0]
        val_samp.index = np.arange(0, val_samp.nsamp)
        val_samp.nfeat = val_samp.x.shape[1]

        val_samp.xmin = val_samp.x.min(0, initial=-self.max_allow_x)
        val_samp.xmax = val_samp.x.max(0, initial=self.max_allow_x)
        val_samp.ymin = val_samp.y.min(initial=-self.max_allow_y)
        val_samp.ymax = val_samp.y.max(initial=self.max_allow_y)

        return trn_samp, val_samp

    def random_selection(self,
                         num=10):

        """
        Method to select a smaller number of samples from the Samples object
        :param num: Number of samples to select
        :return: Samples object
        """

        if num >= self.index.shape[0]:
            print('Number larger than population: {} specified for {} samples'.format(str(num),
                                                                                      str(len(self.index))))
            ran_samp_n = self.index
        else:
            ran_samp_n = np.random.choice(self.index,
                                          size=num,
                                          replace=False)

        # training sample object
        ran_samp = Samples()
        ran_samp.x_name = self.x_name
        ran_samp.y_name = self.y_name
        ran_samp.x = self.x[ran_samp_n, :]
        ran_samp.y = self.y[ran_samp_n]
        ran_samp.nsamp = self.x.shape[0]
        ran_samp.nfeat = self.x.shape[1]
        ran_samp.index = np.arange(0, ran_samp.nsamp)

        ran_samp.xmin = self.x.min(0, initial=-self.max_allow_x)
        ran_samp.xmax = self.x.max(0, initial=self.max_allow_x)

        ran_samp.ymin = self.y.min(initial=-self.max_allow_y)
        ran_samp.ymax = self.y.max(initial=self.max_allow_y)

        return ran_samp

    def selection(self,
                  index_list):
        """
        Method to select samples based on an index list
        :param index_list:
        :return: Samples object
        """
        if type(index_list).__name__ in ('list', 'tuple', 'NoneType'):
            index_list = np.array(Opt.__copy__(index_list))

        samp = Samples()
        samp.x_name = self.x_name
        samp.y_name = self.y_name
        samp.x = self.x[index_list, :]
        samp.y = self.y[index_list]
        samp.nsamp = self.x.shape[0]
        samp.nfeat = self.x.shape[1]
        samp.index = np.arange(0, samp.nsamp)

        samp.xmin = self.x.min(0, initial=-self.max_allow_x)
        samp.xmax = self.x.max(0, initial=self.max_allow_x)

        samp.ymin = self.y.min(initial=-self.max_allow_y)
        samp.ymax = self.y.max(initial=self.max_allow_y)

        return samp

    def add_samp(self,
                 samp):
        """
        merge a Samples object into another
        :param samp:
        :return: None
        """
        self.merge_data(samp)

    def make_folds(self,
                   n_folds=5):

        """
        Make n folds in sample sets
        :param n_folds:
        :return: list of tuples [(training samp, validation samp)...]
        """

        nsamp_list = list(self.index.shape[0] // n_folds for _ in range(n_folds))
        if self.index.shape[0] % n_folds > 0:
            nsamp_list[-1] += self.index.shape[0] % n_folds

        index_list = self.index.copy()
        fold_samples = list()

        for fold_samp in nsamp_list:

            val_index = np.random.choice(index_list,
                                         size=fold_samp,
                                         replace=False)

            trn_index = self.index[~np.in1d(self.index, val_index)]
            index_list = index_list[~np.in1d(index_list, val_index)]

            fold_samples.append((self.selection(trn_index), self.selection(val_index)))

        return fold_samples
