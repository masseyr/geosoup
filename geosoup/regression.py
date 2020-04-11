import pickle
import warnings
from scipy import stats
from math import sqrt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from geosoup.common import Handler, Opt
from geosoup.raster import Raster, np, gdal_array, gdal
from geosoup.common import Timer


__all__ = ['HRFRegressor',
           'RFRegressor',
           'MRegressor']

sep = Handler().sep


class _Regressor(object):
    """
    Regressor base class
    """
    time_it = False

    defaults = {
        'tile_size': 512,
        'n_tile_max': 5,
        'uncert_dict': None,
        'array_additive': 0.,
        'array_multiplier': 1.,
        'nodatavalue': None,
        'out_nodatavalue': None,
        'mask_band': None
    }

    def __init__(self,
                 data=None,
                 regressor=None,
                 **kwargs):
        self.data = data
        self.vdata = None
        self.regressor = regressor
        self.features = None
        self.feature_index = None
        self.label = None
        self.output = None
        self.training_results = dict()
        self.fit = False
        self.all_cv_results = None

        self.adjustment = dict()

        if kwargs is not None:
            if 'timer' in kwargs:
                _Regressor.time_it = kwargs['timer']

    def __repr__(self):
        return "<Regressor base class>"

    def fit_data(self,
                 data,
                 use_weights=False):
        """
        Train the regressor
        :param data: dictionary with values (generated using Samples.format_data())
        :param use_weights: If the sample weights provided should be used? (default: False)
        :return: Nonetype
        """
        self.data = data

        if self.regressor is not None:

            if 'weights' not in data or not use_weights:
                self.regressor.fit(data['features'], data['labels'])
            else:
                self.regressor.fit(data['features'], data['labels'], data['weights'])

        self.features = data['feature_names']
        self.label = data['label_name']
        self.fit = True

        if hasattr(self, 'intercept'):
            if hasattr(self.regressor, 'intercept_'):
                self.intercept = self.regressor.intercept_

        if hasattr(self, 'coefficient'):
            if hasattr(self.regressor, 'coef_'):
                self.coefficient = self.regressor.coef_

    def predict(self, *args, **kwargs):
        """Placeholder function"""
        return

    def pickle_it(self,
                  outfile):
        """
        Save regressor
        :param outfile: File to save the regressor to
        """
        outfile = Handler(filename=outfile).file_remove_check()
        with open(outfile, 'wb') as fileptr:
            pickle.dump(self, fileptr)

    @classmethod
    def load_from_pickle(cls,
                         infile):
        """
        Reload regressor from file
        :param infile: File to load regressor from
        """
        with open(infile, 'rb') as fileptr:
            regressor_obj = pickle.load(fileptr)
            return regressor_obj

    @staticmethod
    @Timer.timing(time_it)
    def regress_raster(regressor,
                       raster_obj,
                       outfile=None,
                       outdir=None,
                       band_name='prediction',
                       output_type='median',
                       **kwargs):

        """Tree variance from the RF regressor
        :param regressor: _Regressor object
        :param raster_obj: Initialized Raster object
        :param outfile: name of output file
        :param band_name: Name of the output raster band
        :param outdir: output folder
        :param output_type: standard deviation ('sd'),
                            variance ('var'),
                            median ('median'),
                            mean ('mean')
                            or confidence interval ('conf')

        :param kwargs:
                        array_multiplier: rescale all band arrays using this value
                        array_additive: add this value to all band arrays
                        out_data_type: output raster data type (GDAL data type)
                        nodatavalue: No data value for input raster
                        out_nodatavalue: Value to replace the input no-data value in output
                        tile_size: Number of pixels in each raster tile
                        band_additives: Values to add to each band array
                        band_multipliers: Values to scale each band array with
                        mask_band: Band to mask the pixels used in regression
                        verbose: If the steps std output should be displayed
                        uncert_dict: Dictionary with each key value pair specifying
                                     a feature band and its uncertainty band. Only
                                     one uncertainty band per feature is allowed.


        :returns: Output as raster object
        """
        if not raster_obj.init:
            raster_obj.initialize()

        defaults = Opt.__copy__(_Regressor.defaults)

        nbands = raster_obj.shape[0]
        nrows = raster_obj.shape[1]
        ncols = raster_obj.shape[2]

        if nbands < len(regressor.features):
            raise ValueError('Regressor trained using more features than supplied raster')
        elif any([feat not in raster_obj.bnames for feat in regressor.features]):
            raise ValueError('Missing trained feature in Raster object')

        kwargs_computed = {
            'out_data_type': gdal_array.NumericTypeCodeToGDALTypeCode(regressor.data['labels'].dtype),
            'tile_size': min([nrows, ncols]) ** 2 if (min([nrows, ncols]) ** 2) <= defaults['tile_size']
            else defaults['tile_size'],
            'band_multipliers': np.array([defaults['array_multiplier'] for _ in raster_obj.bnames]),
            'band_additives': np.array([defaults['array_additive'] for _ in raster_obj.bnames])
        }

        defaults.update(kwargs_computed)
        defaults.update(kwargs)

        if 'array_multiplier' in kwargs:
            if 'band_multipliers' in kwargs:
                defaults['band_multipliers'] = np.array([defaults['band_multipliers'][elem]
                                                        if elem in defaults['band_multipliers']
                                                        else defaults['array_multiplier']
                                                        for elem in raster_obj.bnames])
            else:
                defaults['band_multipliers'] = np.array([defaults['array_multiplier']
                                                         for _ in raster_obj.bnames])

        if 'array_additive' in kwargs:
            if 'band_additives' in kwargs:
                defaults['band_additives'] = np.array([defaults['band_additives'][elem]
                                                      if elem in defaults['band_additives']
                                                      else defaults['array_additive']
                                                      for elem in raster_obj.bnames])
            else:
                defaults['band_additives'] = np.array([defaults['array_additive']
                                                       for _ in raster_obj.bnames])

        if defaults['mask_band'] is not None:
            if type(defaults['mask_band']) == str:
                try:
                    defaults['mask_band'] = raster_obj.bnames.index(defaults['mask_band'])
                except ValueError:
                    warnings.warn('Mask band ignored: Unrecognized band name.')
                    defaults['mask_band'] = None
            elif type(defaults['mask_band']) in (int, float):
                if defaults['mask_band'] > raster_obj.shape[0]:
                    warnings.warn('Mask band ignored: Mask band index greater than number of bands. ' +
                                  'Indices start at 0.')
            else:
                warnings.warn('Mask band ignored: Unrecognized data type.')
                defaults['mask_band'] = None

        # file handler object
        handler = Handler(raster_obj.name)

        # resolving output name
        if outdir is None:
            if outfile is None:
                outfile = handler.add_to_filename('_{}'.format(band_name))
        elif outfile is None:
            handler.dirname = outdir
            outfile = handler.add_to_filename('_{}'.format(band_name))
        else:
            outfile = Handler(outfile).file_remove_check()

        if regressor.feature_index is None:
            regressor.feature_index = list(raster_obj.bnames.index(feat) for feat in regressor.features)

        out_ras_arr = np.zeros([nrows, ncols],
                               dtype=gdal_array.GDALTypeCodeToNumericTypeCode(defaults['out_data_type']))

        raster_obj.make_tile_grid(defaults['tile_size'],
                                  defaults['tile_size'])

        if defaults['verbose']:
            Opt.cprint('\nProcessing {} raster tiles...\n'.format(str(raster_obj.ntiles)))

        count = 0
        for _, tile_arr in raster_obj.get_next_tile():
            tiept_x, tiept_y, tile_cols, tile_rows = raster_obj.tile_grid[count]['block_coords']

            if defaults['verbose']:
                Opt.cprint("\nProcessing tile {} of {}: x {}, y {}, cols {}, rows {}".format(str(count + 1),
                                                                                             str(raster_obj.ntiles),
                                                                                             str(tiept_x),
                                                                                             str(tiept_y),
                                                                                             str(tile_cols),
                                                                                             str(tile_rows)),
                           newline='\n')

            new_shape = [nbands, tile_rows * tile_cols]

            tile_arr = tile_arr.reshape(new_shape)
            tile_arr = tile_arr.swapaxes(0, 1)

            minmax = np.zeros([nbands, 2])
            bad_tile_flag = False

            for ii in range(tile_arr.shape[1]):
                minmax[ii, :] = np.array([np.min(tile_arr[:, ii]), np.max(tile_arr[:, ii])])
                if defaults['verbose']:
                    Opt.cprint('Band {} : '.format(str(ii + 1)) +
                               'min {} max {}'.format(str(minmax[ii, 0]),
                                                      str(minmax[ii, 1])))

                if defaults['nodatavalue'] is not None:
                    if (minmax[ii, 0] == minmax[ii, 1] == defaults['nodatavalue']) \
                            and (ii in regressor.feature_index):

                        bad_tile_flag = True

            if not bad_tile_flag:
                for k, v in defaults.items():
                    print('{} : {}'.format(str(k), str(v)))
                for elem in raster_obj.tile_grid:
                    print(elem)
                tile_arr_out = regressor.predict(tile_arr,
                                                 output_type=output_type,
                                                 **defaults)
            else:
                tile_arr_out = np.zeros([tile_rows * tile_cols]) + defaults['out_nodatavalue']

            if tile_arr_out.dtype != out_ras_arr.dtype:
                tile_arr_out = tile_arr_out.astype(out_ras_arr.dtype)

            if defaults['mask_band'] is not None:
                tile_arr_out_reshaped = tile_arr_out.reshape([tile_rows, tile_cols]) * \
                                        tile_arr[defaults['mask_band'], :, :]
            else:
                tile_arr_out_reshaped = tile_arr_out.reshape([tile_rows, tile_cols])

            out_ras_arr[tiept_y: (tiept_y + tile_rows), tiept_x: (tiept_x + tile_cols)] = tile_arr_out_reshaped

            count += 1

        if defaults['verbose']:
            Opt.cprint("\nInternal tile processing completed\n")

        out_ras = Raster(outfile)
        out_ras.dtype = defaults['out_data_type']
        out_ras.transform = raster_obj.transform
        out_ras.crs_string = raster_obj.crs_string

        out_ras.array = out_ras_arr
        out_ras.shape = [1, nrows, ncols]
        out_ras.bnames = [band_name]
        out_ras.nodatavalue = defaults['out_nodatavalue']

        # return raster object
        return out_ras

    @staticmethod
    def linear_regress(x,
                       y,
                       xlim=None,
                       ylim=None):
        """
        Calculate linear regression attributes
        :param x: Vector of independent variables 1D
        :param y: Vector of dependent variables 1D
        :param xlim: 2 element list or tuple [lower limit, upper limit]
        :param ylim: 2 element list or tuple [lower limit, upper limit]
        """
        if type(x).__name__ in ('list', 'tuple', 'NoneType'):
            x_ = np.array(x)
        else:
            x_ = x.copy()

        if type(y).__name__ in ('list', 'tuple', 'NoneType'):
            y_ = np.array(y)
        else:
            y_ = y.copy()

        if xlim is not None:
            y_ = y_[np.where((x_ >= xlim[0]) & (x_ <= xlim[1]))]
            x_ = x_[np.where((x_ >= xlim[0]) & (x_ <= xlim[1]))]

        if ylim is not None:
            x_ = x_[np.where((y_ >= ylim[0]) & (y_ <= ylim[1]))]
            y_ = y_[np.where((y_ >= ylim[0]) & (y_ <= ylim[1]))]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_, y_)
        rsq = r_value ** 2

        out_dict = {
            'rsq': rsq,
            'slope': slope,
            'intercept': intercept,
            'pval': p_value,
            'stderr': std_err
        }

        return out_dict

    @staticmethod
    @Timer.timing(True)
    def fit_regressor(args_list,
                      rank=None):
        """
        Method to train and validate classification models using MPI
        and if the R-squared is > 0.5 then store the model and its
        properties in a pickled file and csv file respectively

        :param args_list: List of MPI scattered samples. Each element
                        in the args_list includes a list in the given order of:
                                name: Name of the model,
                                train_samp: Samples object for training the classifier,
                                valid_samp: Samples object for validating the classifier,
                                in_file: input file containing the samples
                                pickle_dir: folder to store the pickled classifier in)
                                llim: lower limit
                                ulim: upper limit
                                param: RFRegressor initialization parameters

        :param rank: rank of the MPI node/processor

        :returns: tuple (r-squared*100 , model_name)
        """
        sep = Handler().sep

        result_list = list()

        Opt.cprint('Length of arguments at {} is {}'.format(str(rank), len(args_list)))

        if args_list is None:
            args_list = list()

        for args in args_list:

            name, train_samp, valid_samp, in_file, pickle_dir, llim, ulim, param = args

            # initialize RF classifier
            model = RFRegressor(**param)
            model.time_it = True

            regress_limit = [0.025 * ulim, 0.975 * ulim]
            rsq_limit = 60.0

            # fit RF classifier using training data
            model.fit_data(train_samp.format_data())

            # predict using held out samples and print to file
            pred = model.sample_predictions(valid_samp.format_data(),
                                            regress_limit=regress_limit)

            out_dict = dict()
            out_dict['name'] = Handler(in_file).basename.split('.')[0] + name
            out_dict['rsq'] = pred['rsq'] * 100.0
            out_dict['slope'] = pred['slope']
            out_dict['intercept'] = pred['intercept']
            out_dict['rmse'] = pred['rmse']

            model.output = out_dict

            rsq = pred['rsq'] * 100.0
            slope = pred['slope']
            intercept = pred['intercept']
            rmse = pred['rmse']

            if rsq >= rsq_limit:
                if intercept > regress_limit[0]:
                    model.adjustment['bias'] = -1.0 * (intercept / slope)

                model.adjustment['gain'] = 1.0 / slope
                model.adjustment['upper_limit'] = ulim
                model.adjustment['lower_limit'] = llim

                # file to write the model run output to
                outfile = pickle_dir + sep + \
                    Handler(in_file).basename.split('.')[0] + name + '.txt'
                outfile = Handler(filename=outfile).file_remove_check()

                # save RF classifier using pickle
                picklefile = pickle_dir + sep + \
                    Handler(in_file).basename.split('.')[0] + name + '.pickle'
                picklefile = Handler(filename=picklefile).file_remove_check()

                # predict using the model to store results in a file
                pred = model.sample_predictions(valid_samp.format_data(),
                                                outfile=outfile,
                                                picklefile=picklefile,
                                                regress_limit=regress_limit)

                out_dict['rsq'] = pred['rsq'] * 100.0
                out_dict['slope'] = pred['slope']
                out_dict['intercept'] = pred['intercept']
                out_dict['rmse'] = pred['rmse']

                out_dict['regress_low_limit'] = regress_limit[0]
                out_dict['regress_up_limit'] = regress_limit[1]

                model.output.update(out_dict)
                model.pickle_it(picklefile)

            result_list.append(out_dict)

        return result_list


class MRegressor(_Regressor):
    """Multiple linear regressor
    This uses scikit-learn multiple regressor library
    """

    time_it = False

    def __init__(self,
                 data=None,
                 regressor=None,
                 intercept=True,
                 jobs=1,
                 normalize=False,
                 **kwargs):

        super(MRegressor, self).__init__(data,
                                         regressor)
        if self.regressor is None:
            self.regressor = linear_model.LinearRegression(copy_X=True,
                                                           fit_intercept=intercept,
                                                           n_jobs=jobs,
                                                           normalize=normalize)

        self.intercept = self.regressor.intercept_ if hasattr(self.regressor, 'intercept_') else None
        self.coefficient = self.regressor.coef_ if hasattr(self.regressor, 'coef_') else None

        if kwargs is not None:
            if 'timer' in kwargs:
                MRegressor.time_it = kwargs['timer']

    def __repr__(self):
        # gather which attributes exist
        attr_truth = [self.coefficient is not None,
                      self.intercept is not None]

        if any(attr_truth):

            print_str_list = list("Multiple Linear Regressor:\n")

            # strings to be printed for each attribute
            if attr_truth[0]:
                print_str_list.append("Coefficients: {}\n".format(', '.join([str(elem) for elem in
                                                                             self.coefficient.tolist()])))

            if attr_truth[1]:
                print_str_list.append("Intercept: {}\n".format(self.intercept))

            # combine all strings into one print string
            print_str = ''.join(print_str_list)

            return print_str

        else:
            # if empty return empty
            return "<Multiple Linear Regressor: __empty__>"

    @Timer.timing(time_it)
    def predict(self,
                arr,
                ntile_max=5,
                tile_size=1024,
                **kwargs):
        """
        Calculate multiple regression model prediction, variance, or standard deviation.
        Variance or standard deviation is calculated across all trees.
        Tiling is necessary in this step because large numpy arrays can cause
        memory issues during creation.

        :param arr: input numpy 2d array (axis 0: features (pixels), axis 1: bands)
        :param ntile_max: Maximum number of tiles up to which the
                          input image or array is processed without tiling (default = 9).
                          You can choose any (small) number that suits the available memory.
        :param tile_size: Size of each square tile (default = 128)
                :param kwargs: Keyword arguments:
                       'gain': Adjustment of the predicted output by linear adjustment of gain (slope)
                       'bias': Adjustment of the predicted output by linear adjustment of bias (intercept)
                       'upper_limit': Limit of maximum value of prediction
                       'lower_limit': Limit of minimum value of prediction
        :return: 1d image array (that will need reshaping if image output)
        """
        nodatavalue = None
        verbose = False

        if kwargs is not None:
            for key, value in kwargs.items():
                if key in ('gain', 'bias', 'upper_limit', 'lower_limit'):
                    self.adjustment[key] = value

                if key == 'nodatavalue':
                    nodatavalue = value

                if key == 'verbose':
                    verbose = value

        if type(arr).__name__ != 'ndarray':
            arr = np.array(arr)

        # define output array
        out_arr = np.zeros(arr.shape[0])

        # input image size
        npx_inp = int(arr.shape[0])  # number of pixels in input image
        nb_inp = int(arr.shape[1])  # number of bands in input image

        # size of tiles
        npx_tile = int(tile_size)  # pixels in each tile
        npx_last = npx_inp % npx_tile  # pixels in last tile
        ntiles = (npx_inp // npx_tile) + 1  # total tiles

        # if number of tiles in the image
        # are less than the specified number
        if ntiles > ntile_max:

            for i in range(0, ntiles - 1):
                if verbose:
                    Opt.cprint('Processing tile {} of {}'.format(str(i+1), ntiles))

                # calculate predictions for each pixel in a 2d array
                out_arr[i * npx_tile:(i + 1) * npx_tile] = \
                    self.regressor.predict(arr[i * npx_tile:(i + 1) * npx_tile, self.feature_index])

            if npx_last > 0:  # number of total pixels for the last tile

                i = ntiles - 1
                if verbose:
                    Opt.cprint('Processing tile {} of {}'.format(str(i+1), ntiles))
                out_arr[i * npx_last:(i + 1) * npx_last] = \
                    self.regressor.predict(arr[i * npx_tile:(i * npx_tile + npx_last), self.feature_index])

        else:
            out_arr = self.regressor.predict(arr[:, self.feature_index])

        if len(self.adjustment) > 0:

            if 'gain' in self.adjustment:
                out_arr = out_arr * self.adjustment['gain']

            if 'bias' in self.adjustment:
                out_arr = out_arr + self.adjustment['bias']

            if 'upper_limit' in self.adjustment:
                out_arr[out_arr > self.adjustment['upper_limit']] = self.adjustment['upper_limit']

            if 'lower_limit' in self.adjustment:
                out_arr[out_arr < self.adjustment['lower_limit']] = self.adjustment['lower_limit']

        output = out_arr

        if nodatavalue is not None:
            for ii in range(arr.shape[0]):
                output[np.unique(np.where(arr[ii, :, :] == nodatavalue)[0])] = nodatavalue

        return output

    def sample_predictions(self,
                           data,
                           picklefile=None,
                           outfile=None,
                           **kwargs):
        """
        Get predictions from the multiple regressor
        :param data: Dictionary object from Samples.format_data
        :param picklefile: Random Forest pickle file
        :param outfile: output csv file name
        """
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False

        self.feature_index = list(data['feature_names'].index(feat) for feat in self.features)

        # calculate variance of tree predictions
        y = self.predict(data['features'],
                         verbose=verbose)

        # rms error of the predicted versus actual
        rmse = sqrt(mean_squared_error(data['labels'], y))

        # r-squared of predicted versus actual
        lm = self.linear_regress(data['labels'], y)

        # if either one of outfile or pickle file are available
        # then raise error
        if (outfile is not None) != (picklefile is not None):
            raise ValueError("Missing outfile or picklefile")

        # if outfile and pickle file are both available
        # then write to file and proceed to return
        elif outfile is not None:
            # write y, y_hat_bar, var_y to file (<- rows in this order)
            y_list = []
            y_list += y.tolist()
            out_list = ['obs_y,' + ', '.join([str(elem) for elem in data['labels'].tolist()]),
                        'pred_y,' + ', '.join([str(elem) for elem in y_list]),
                        'rmse,' + str(rmse),
                        'rsq,' + str(lm['rsq']),
                        'slope,' + str(lm['slope']),
                        'intercept,' + str(lm['intercept']),
                        'rf_file,' + picklefile]

            # write the list to file
            Handler(filename=outfile).write_list_to_file(out_list)

        # if outfile and pickle file are not provided,
        # then only return values
        return {
            'pred_y': y,
            'obs_y': data['labels'],
            'rmse': rmse,
            'rsq': lm['rsq'],
            'slope': lm['slope'],
            'intercept': lm['intercept'],
            'model': {'intercept': self.intercept, 'coefficient': self.coefficient}
        }

    def get_training_fit(self,
                         regress_limit=None):

        """
        Find out how well the training samples fit the model
        :param regress_limit: List of upper and lower regression limits for training fit prediction
        :return: None
        """
        if self.fit:
            # predict using held out samples and print to file
            pred = self.sample_predictions(self.data,
                                           regress_limit=regress_limit)

            self.training_results['rsq'] = pred['rsq'] * 100.0
            self.training_results['slope'] = pred['slope']
            self.training_results['intercept'] = pred['intercept']
            self.training_results['rmse'] = pred['rmse']
        else:
            raise ValueError("Model not initialized with samples")

    def get_adjustment_param(self,
                             clip=0.025,
                             data_limits=None,
                             over_adjust=1.0):
        """
        get the model adjustment parameters based on training fit
        :param clip: ratio of the data to be clipped from either ends to fit a constraining regression
        :param data_limits: minimum and maximum value of the output, tuple
        :param over_adjust: factor to multiply the final output with

        :return: None
        """
        if data_limits is None:
            data_limits = [self.data['labels'].min(), self.data['labels'].max()]

        regress_limit = [data_limits[0] + clip * (data_limits[1]-data_limits[0]),
                         data_limits[1] - clip * (data_limits[1]-data_limits[0])]

        self.get_training_fit(regress_limit=regress_limit)

        if self.training_results['intercept'] > regress_limit[0]:
            self.adjustment['bias'] = -1.0 * (self.training_results['intercept'] / self.training_results['slope'])

        self.adjustment['gain'] = (1.0 / self.training_results['slope']) * over_adjust

        self.adjustment['lower_limit'] = data_limits[0]
        self.adjustment['upper_limit'] = data_limits[1]


class RFRegressor(_Regressor):
    """Random Forest Regressor.
     This uses scikit-learn Random Forest regressor"""

    time_it = False

    def __init__(self,
                 data=None,
                 regressor=None,
                 trees=10,
                 samp_split=2,
                 samp_leaf=1,
                 max_depth=None,
                 max_feat='auto',
                 oob_score=False,
                 criterion='mse',
                 n_jobs=1,
                 **kwargs):
        """
        Initialize RF regressor using class parameters
        :param trees: Number of trees
        :param samp_split: Minimum number of samples for split
        :param oob_score: (bool) calculate out of bag score
        :param criterion: criterion to be used (default: 'mse', options: 'mse', 'mae')
        (some parameters are as detailed in
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
        """

        super(RFRegressor, self).__init__(data,
                                          regressor)

        if self.regressor is None:
            self.regressor = RandomForestRegressor(n_estimators=trees,
                                                   max_depth=max_depth,
                                                   min_samples_split=samp_split,
                                                   min_samples_leaf=samp_leaf,
                                                   max_features=max_feat,
                                                   criterion=criterion,
                                                   oob_score=oob_score,
                                                   n_jobs=n_jobs)
        self.trees = trees
        self.max_depth = max_depth
        self.samp_split = samp_split
        self.samp_leaf = samp_leaf
        self.max_feat = max_feat
        self.oob_score = oob_score
        self.criterion = criterion
        self.n_jobs = n_jobs

        self.dec_paths = list()

        if kwargs is not None:
            if 'timer' in kwargs:
                MRegressor.time_it = kwargs['timer']

    def __repr__(self):
        # gather which attributes exist
        attr_truth = [hasattr(self.regressor, 'estimators_'),
                      hasattr(self.regressor, 'n_features_'),
                      hasattr(self.regressor, 'n_outputs_'),
                      hasattr(self.regressor, 'oob_score_')]

        # if any exist print them
        if any(attr_truth):

            print_str_list = list("Random Forest Regressor:\n")

            # strings to be printed for each attribute
            if attr_truth[0]:
                print_str_list.append("Estimators: {}\n".format(len(self.regressor.estimators_)))

            if attr_truth[1]:
                print_str_list.append("Features: {} : {} \n".format(self.regressor.n_features_,
                                                                    ', '.join(self.features)))

            if attr_truth[2]:
                print_str_list.append("Output: {}\n".format(self.regressor.n_outputs_))

            if attr_truth[3]:
                print_str_list.append("OOB Score: {:{w}.{p}f} %".format(self.regressor.oob_score_ * 100.0,
                                                                        w=3, p=2))

            # combine all strings into one print string
            print_str = ''.join(print_str_list)

            return print_str

        else:
            # if empty return empty
            return "<Random Forest Regressor: __empty__>"

    def regress_tile(self,
                     arr,
                     tile_start=None,
                     tile_end=None,
                     output_type='median',
                     nodatavalue=None,
                     intvl=None,
                     min_variance=None,
                     **kwargs):

        """
        Method to regress each tile of the image using one RF regressor
        :param arr: input 2D array to process (rows = elements, columns = features)
        :param tile_start: pixel location of tile start
        :param tile_end: pixel location of tile end
        :param output_type: Type of output to produce,
               choices: ['sd', 'var', 'full', 'mean', 'median']
               where 'sd' is for standard deviation,
               'var' is for variance
               'full is for all leaf outputs
               'median' is for median of tree outputs
               'mean' is for mean of tree outputs
        :param nodatavalue: No data value
        :param intvl: Prediction interval width (default: 95 percentile)
        :param min_variance: Minimum variance after which to cutoff
        :return: numpy 1-D array
        """
        if min_variance is None:
            if intvl is not None:
                min_variance = (1 - intvl/100.) * np.min(arr.astype(np.float32))
            else:
                min_variance = 0.05 * np.min(arr.astype(np.float32))

        if tile_end is None:
            tile_end = arr.shape[0]

        if tile_start is None:
            tile_start = 0

        # List of index of bands to be used for regression
        if 'feature_index' in kwargs:
            feature_index = np.array(kwargs['feature_index'])
        elif self.feature_index is not None:
            feature_index = np.array(self.feature_index)
        else:
            feature_index = np.array(range(0, arr.shape[0]))

        band_multipliers = np.repeat(1.0, feature_index.shape[0]) if 'band_multipliers' not in kwargs \
            else kwargs['band_multipliers']

        band_additives = np.repeat(0.0, feature_index.shape[0]) if 'band_additives' not in kwargs \
            else kwargs['band_additives']

        feat_arr = arr[tile_start:tile_end, feature_index] * \
            band_multipliers[feature_index] + \
            band_additives[feature_index]

        if nodatavalue is not None:
            mask_arr = np.apply_along_axis(lambda x: 0
                                           if np.any(x == nodatavalue) else 1,
                                           0,
                                           feat_arr)
        else:
            mask_arr = np.zeros([feat_arr.shape[1]]) + 1

        out_tile = np.zeros([tile_end - tile_start])
        tile_arr = np.zeros([self.trees, (tile_end - tile_start)])

        if output_type in ('mean', 'median', 'full'):

            # calculate tree predictions for each pixel in a 2d array
            for jj, tree_ in enumerate(self.regressor.estimators_):
                tile_arr[jj, :] = tree_.predict(feat_arr)

            if output_type == 'median':
                out_tile = np.median(tile_arr, axis=0)
            elif output_type == 'mean':
                out_tile = np.mean(tile_arr, axis=0)
            elif output_type == 'full':
                return tile_arr

        elif output_type in ('sd', 'var'):

            for jj, tree_ in enumerate(self.regressor.estimators_):
                tile_arr[jj, :] = tree_.predict(feat_arr)

                var_tree = tree_.tree_.impurity[tree_.apply(feat_arr)]

                var_tree[var_tree < min_variance] = min_variance
                mean_tree = tree_.predict(feat_arr)
                out_tile += var_tree + mean_tree ** 2

            predictions = np.mean(tile_arr, axis=0)

            out_tile /= len(self.regressor.estimators_)
            out_tile -= predictions ** 2.0
            out_tile[out_tile < 0.0] = 0.0

            if output_type == 'sd':
                out_tile = out_tile ** 0.5
        else:
            raise RuntimeError("Unknown output type or no output type specified")

        if len(self.adjustment) > 0:

            if 'gain' in self.adjustment:
                out_tile = out_tile * self.adjustment['gain']

            if 'bias' in self.adjustment:
                out_tile = out_tile + self.adjustment['bias']

            if 'upper_limit' in self.adjustment:
                out_tile[out_tile > self.adjustment['upper_limit']] = self.adjustment['upper_limit']

            if 'lower_limit' in self.adjustment:
                out_tile[out_tile < self.adjustment['lower_limit']] = self.adjustment['lower_limit']

        if nodatavalue is not None:
            out_tile[np.where(mask_arr == 0)] = kwargs['out_nodatavalue']

        return out_tile

    @staticmethod
    def pixel_range(pixel_vec,
                    **kwargs):
        """
        Method to compute range of regression uncertainty for each pixel
        :param pixel_vec: Input pixel vector containing feature and uncertainty bands
        :param kwargs: Additional keyword arguments to be passed on to

                      regressor: RFRegressor object
                      uncert_dict: Dictionary specifying the indices of
                                   feature bands (keys) and their corresponding
                                   uncertainty bands (values)
                      n_rand: Number of random values to generate
                              in the uncertainty range (default: 5)
                      half_range: If the input and output uncertainty values are
                             - full range (x +/- a), or
                             - half range (x +/- a/2)

        :return: range of uncertainty values if one or more uncertainty bands
                 are specified in uncertainty dict else returns 0
        """
        n_rand = kwargs.pop('n_rand', 5)
        uncert_dict = kwargs.pop('uncert_dict', {})
        output_type = kwargs.pop('output_type', 'median')
        n_samp = n_rand ** len(uncert_dict)

        if n_samp > 0:
            feat_arr = np.tile(pixel_vec, (n_samp, 1))

            feat_vals = pixel_vec[uncert_dict.keys()].tolist()
            uncert_vals = pixel_vec[uncert_dict.values()].tolist()

            half_range = kwargs.pop('half_range', True)

            if half_range:
                uncert_rand_lists = list((np.random.rand(n_rand) - 0.5) * uncert_val + feat_vals[ii] for
                                         ii, uncert_val in enumerate(uncert_vals))
            else:
                uncert_rand_lists = list((2 * np.random.rand(n_rand) - 1) * uncert_val + feat_vals[ii] for
                                         ii, uncert_val in enumerate(uncert_vals))

            feat_arr[:, uncert_dict.keys()] = np.array(zip(*list(np.array(temp_arr).flatten()
                                                                 for temp_arr in
                                                                 np.meshgrid(*uncert_rand_lists))))

            regressor = kwargs.pop('regressor')

            pred_arr = regressor.regress_tile(feat_arr,
                                              tile_start=0,
                                              tile_end=n_samp,
                                              output_type=output_type,
                                              **kwargs).flatten()

            return np.abs(pred_arr.min() - pred_arr.max())
        else:
            return

    def regress_tile_uncert(self,
                            arr,
                            tile_start=None,
                            tile_end=None,
                            output_type='median',
                            uncert_dict=None,
                            n_rand=5,
                            half_range=True,
                            compare_uncert=False,
                            calculated_uncert_type='sd',
                            **kwargs):
        """
        Method to regress each tile of the image and compute range of uncertainty values from one RF regressor
        :param arr: input 2D array to process (rows = elements, columns = features)
        :param tile_start: pixel location of tile start
        :param tile_end: pixel location of tile end
        :param output_type: Type of output to produce,
               choices: ['sd', 'var', 'full', 'mean', 'median']
               where 'sd' is for standard deviation,
               'var' is for variance
               'full is for all leaf outputs
               'median' is for median of tree outputs
               'mean' is for mean of tree outputs
        :param uncert_dict: Dictionary specifying the indices of
                            feature bands (keys) and their corresponding
                            uncertainty bands (values)
        :param n_rand: Number of random values to generate in the uncertainty range (default: 5)
        :param half_range: If the input and output uncertainty values are
                             - full range (x +/- a), or
                             - half range (x +/- a/2)
        :param compare_uncert: Boolean. Compare the propagated uncertainty with
                              uncertainty in RF regression output value
                              and return the larger of the two (default: False)
        :param calculated_uncert_type: Type of value to compute as uncertainty of prediction
        :param kwargs: Keyword arguments:
                ntile_max: Maximum number of tiles up to which the
                           input image or array is processed without tiling (default = 9).
                           You can choose any (small) number that suits the available memory.
                tile_size: Number of pixels in each tile (default = 1024)
                gain: Adjustment of the predicted output by linear adjustment of gain (slope)
                bias: Adjustment of the predicted output by linear adjustment of bias (intercept)
                upper_limit: Limit of maximum value of prediction
                lower_limit: Limit of minimum value of prediction
                intvl: Prediction interval width (default: 95 percentile)
                uncert_dict: Dictionary specifying the indices of
                           feature bands (keys) and their corresponding
                           uncertainty bands (values)
                n_rand: Number of random values to generate in the uncertainty range (default: 5)
                half_range (Boolean): If the input and output uncertainty values are
                           False  - full range (x +/- a), or
                           True   - half range (x +/- a/2)
        """
        uncert_kwargs = Opt.__copy__(kwargs)
        if uncert_dict is not None:

            uncert_kwargs.update({'regressor': self,
                                  'uncert_dict': uncert_dict,
                                  'n_rand': n_rand,
                                  'output_type': output_type,
                                  'half_range': half_range})

            propagated_uncert = np.apply_along_axis(self.pixel_range,
                                                    1,
                                                    arr[tile_start:tile_end, :],
                                                    **uncert_kwargs)

            if compare_uncert:
                calculated_uncert = self.regress_tile(arr,
                                                      tile_start,
                                                      tile_end,
                                                      output_type=calculated_uncert_type,
                                                      **kwargs)

                return np.apply_along_axis(lambda x: np.max(x),
                                           0,
                                           np.array([propagated_uncert, calculated_uncert]))
            else:
                return propagated_uncert
        else:
            return

    @Timer.timing(time_it)
    def predict(self,
                arr,
                output_type='median',
                **kwargs):
        """
        Calculate random forest model prediction, variance, or standard deviation.
        Variance or standard deviation is calculated across all trees.
        Tiling is necessary in this step because large numpy arrays can cause
        memory issues during creation.

        :param arr: input 2d array (axis 0: features (pixels), axis 1: bands)

        :param output_type: which output to produce,
                       choices: ['sd', 'var', 'median', 'mean', 'full']
                       where 'sd' is for standard deviation,
                       'var' is for variance
                       'median' is for median of tree outputs
                       'mean' is for mean of tree outputs
                       'full' is for the full spectrum of the leaf nodes' prediction

        :param kwargs: Keyword arguments:
                        ntile_max: Maximum number of tiles up to which the
                                   input image or array is processed without tiling (default = 5).
                                   You can choose any (small) number that suits the available memory.
                        tile_size: Number of pixels in each tile (default = 102400)
                        gain: Adjustment of the predicted output by linear adjustment of gain (slope)
                        bias: Adjustment of the predicted output by linear adjustment of bias (intercept)
                        upper_limit: Limit of maximum value of prediction
                        lower_limit: Limit of minimum value of prediction
                        intvl: Prediction interval width (default: 95 percentile)
                        uncert_dict: Dictionary specifying the indices of
                                   feature bands (keys) and their corresponding
                                   uncertainty bands (values)
                        n_rand: Number of random values to generate in the uncertainty range (default: 5)
                        half_range (Boolean): If the input and output uncertainty values are
                                   False  - full range (x +/- a), or
                                   True   - half range (x +/- a/2)

        :return: 1d image array (that will need reshaping if image output)
        """
        if not type(arr) == np.ndarray:
            arr = np.array(arr)

        defaults = Opt.__copy__(self.defaults)
        defaults.update(kwargs)

        uncert_dict = defaults['uncert_dict']
        tile_size = defaults['tile_size'] ** 2
        n_tile_max = defaults['n_tile_max']

        print('tile_size: -- {}'.format(tile_size))

        for key, value in defaults.items():
            if key in ('gain', 'bias', 'upper_limit', 'lower_limit'):
                if defaults[key] is not None:
                    self.adjustment[key] = value

        # define output array
        if output_type == 'full':
            out_arr = np.zeros([self.trees, arr.shape[0]])
        else:
            out_arr = np.zeros(arr.shape[0])

        # input image size
        npx_inp = int(arr.shape[0])  # number of pixels in input image
        nb_inp = int(arr.shape[1])  # number of bands in input image

        # size of tiles
        npx_tile = int(tile_size)  # pixels in each tile
        npx_last = npx_inp % npx_tile  # pixels in last tile
        ntiles = (npx_inp // npx_tile) + 1  # total tiles

        if ntiles > n_tile_max:

            for i in range(0, ntiles - 1):
                if kwargs['verbose']:
                    Opt.cprint('Processing internal tile {} of {}'.format(str(i+1), ntiles),
                               newline='')

                if output_type == 'full':
                    if uncert_dict is not None and type(uncert_dict) == dict:
                        temp_tile_ = self.regress_tile_uncert(arr,
                                                              i * npx_tile,
                                                              (i + 1) * npx_tile,
                                                              output_type,
                                                              **kwargs)
                    else:
                        temp_tile_ = self.regress_tile(arr,
                                                       i * npx_tile,
                                                       (i + 1) * npx_tile,
                                                       **kwargs)

                    out_arr[:, i * npx_tile:(i + 1) * npx_tile] = temp_tile_
                    if kwargs['verbose']:
                        Opt.cprint(' min {} max {}'.format(np.min(temp_tile_), np.max(temp_tile_)))

                else:
                    if uncert_dict is not None and type(uncert_dict) == dict:
                        temp_tile_ = self.regress_tile_uncert(arr,
                                                              i * npx_tile,
                                                              (i + 1) * npx_tile,
                                                              **kwargs)
                    else:
                        temp_tile_ = self.regress_tile(arr,
                                                       i * npx_tile,
                                                       (i + 1) * npx_tile,
                                                       **kwargs)

                    out_arr[i * npx_tile:(i + 1) * npx_tile] = temp_tile_
                    if kwargs['verbose']:
                        Opt.cprint(' min {} max {}'.format(np.min(temp_tile_), np.max(temp_tile_)))

            if npx_last > 0:  # number of total pixels for the last tile

                i = ntiles - 1
                if kwargs['verbose']:
                    Opt.cprint('Processing internal tile {} of {}'.format(str(i+1), ntiles),
                               newline='')

                if output_type == 'full':
                    if uncert_dict is not None and type(uncert_dict) == dict:
                        temp_tile_ = self.regress_tile_uncert(arr,
                                                              i * npx_tile,
                                                              i * npx_tile + npx_last,
                                                              **kwargs)
                    else:
                        temp_tile_ = self.regress_tile(arr,
                                                       i * npx_tile,
                                                       i * npx_tile + npx_last,
                                                       **kwargs)

                    out_arr[:, i * npx_tile:(i * npx_tile + npx_last)] = temp_tile_
                    if kwargs['verbose']:
                        Opt.cprint(' min {} max {}'.format(np.min(temp_tile_), np.max(temp_tile_)))

                else:
                    if uncert_dict is not None and type(uncert_dict) == dict:
                        temp_tile_ = self.regress_tile_uncert(arr,
                                                              i * npx_tile,
                                                              i * npx_tile + npx_last,
                                                              **kwargs)
                    else:
                        temp_tile_ = self.regress_tile(arr,
                                                       i * npx_tile,
                                                       i * npx_tile + npx_last,
                                                       **kwargs)

                    out_arr[i * npx_tile:(i * npx_tile + npx_last)] = temp_tile_
                    if kwargs['verbose']:
                        Opt.cprint(' min {} max {}'.format(np.min(temp_tile_), np.max(temp_tile_)))

        else:
            if kwargs['verbose']:
                Opt.cprint('Processing internal tile 1 of 1',
                           newline='')
            if uncert_dict is not None and type(uncert_dict) == dict:
                out_arr = self.regress_tile_uncert(arr,
                                                   0,
                                                   npx_inp,
                                                   **kwargs)
            else:
                out_arr = self.regress_tile(arr,
                                            0,
                                            npx_inp,
                                            **kwargs)
            if kwargs['verbose']:
                Opt.cprint(' min {} max {}'.format(np.min(out_arr), np.max(out_arr)))

        return out_arr

    def sample_predictions(self,
                           data,
                           picklefile=None,
                           outfile=None,
                           output='median',
                           **kwargs):
        """
        Get tree predictions from the RF regressor
        :param data: Dictionary object from Samples.format_data
        :param picklefile: Random Forest pickle file
        :param output: Metric to be omputed from the random forest (options: 'mean','median','sd')
        :param outfile: output csv file name
        :param kwargs: Keyword arguments:
               'gain': Adjustment of the predicted output by linear adjustment of gain (slope)
               'bias': Adjustment of the predicted output by linear adjustment of bias (intercept)
               'upper_limit': Limit of maximum value of prediction
               'lower_limit': Limit of minimum value of prediction
               'regress_limit': 2 element list of Minimum and Maximum limits of the label array [min, max]
               'all_y': Boolean (if all leaf outputs should be calculated)
               'var_y': Boolean (if variance of leaf nodes should be calculated)
               'sd_y': Boolean (if the standard dev of all values at a leaf should be calculated)
        """
        for key, value in kwargs.items():
            if key in ('gain', 'bias', 'upper_limit', 'lower_limit'):
                self.adjustment[key] = value

        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False

        self.feature_index = list(data['feature_names'].index(feat) for feat in self.features)

        if 'regress_limit' in kwargs:
            regress_limit = kwargs['regress_limit']
        else:
            regress_limit = None

        var_y = all_y = sd_y = mean = pred_y = None

        # calculate variance of tree predictions
        if 'var_y' in kwargs and kwargs['var_y']:
            var_y = self.predict(data['features'],
                                 output_type='var',
                                 verbose=verbose)

        # calculate mean of tree predictions
        if 'all_y' in kwargs and kwargs['all_y']:
            all_y = self.predict(data['features'],
                                 output_type='full',
                                 verbose=verbose)

        # calculate sd of tree predictions
        if 'sd_y' in kwargs and kwargs['sd_y']:
            sd_y = self.predict(data['features'],
                                output_type='sd',
                                verbose=verbose)

        # calculate sd of tree predictions
        if 'mean' in kwargs and kwargs['se_y']:
            mean = self.predict(data['features'],
                                output_type='mean',
                                verbose=verbose)

        # calculate median tree predictions
        pred_y = self.predict(data['features'],
                              output_type=output,
                              verbose=verbose)

        # rms error of the predicted versus actual
        rmse = sqrt(mean_squared_error(data['labels'], pred_y))

        # r-squared of predicted versus actual
        if regress_limit is not None:
            lm = self.linear_regress(data['labels'],
                                     pred_y,
                                     xlim=regress_limit)
        else:
            lm = self.linear_regress(data['labels'],
                                     pred_y)

        # if either one of outfile or pickle file are available
        # then raise error
        if (outfile is not None) != (picklefile is not None):
            raise ValueError("Missing outfile or picklefile")

        # if outfile and pickle file are both available
        # then write to file and proceed to return
        elif outfile is not None:
            # write y, y_hat_bar, var_y to file (<- rows in this order)
            out_list = ['obs_y,' + ', '.join([str(elem) for elem in data['labels']]),
                        'pred_y,' + ', '.join([str(elem) for elem in pred_y]),
                        'rmse,' + str(rmse),
                        'rsq,' + str(lm['rsq']),
                        'slope,' + str(lm['slope']),
                        'intercept,' + str(lm['intercept']),
                        'rf_file,' + picklefile]

            if all_y is not None:
                out_list.append('all_y,' + '[' + ', '.join(['[' + ', '.join(str(elem) for
                                                            elem in arr) + ']' for arr in list(all_y)]) + ']')
            if var_y is not None:
                out_list.append('var_y,' + ', '.join([str(elem) for elem in var_y]))

            if sd_y is not None:
                out_list.append('sd_y,' + ', '.join([str(elem) for elem in sd_y]))

            if mean is not None:
                out_list.append('mean,' + ', '.join([str(elem) for elem in mean]))

            # write the list to file
            Handler(filename=outfile).write_list_to_file(out_list)

        # if outfile and pickle file are not provided,
        # then only return values
        out_dict = {
            'pred_y': pred_y,
            'obs_y': data['labels'],
            'rmse': rmse,
            'rsq': lm['rsq'],
            'slope': lm['slope'],
            'intercept': lm['intercept'],
        }

        if all_y is not None:
            out_dict['all_y'] = all_y
        if var_y is not None:
            out_dict['var_y'] = var_y
        if sd_y is not None:
            out_dict['sd_y'] = sd_y
        if mean is not None:
            out_dict['mean'] = mean

        return out_dict

    def var_importance(self):
        """
        Return list of tuples of band names and their importance
        """

        return [(band, importance) for band, importance in
                zip(self.data['feature_names'], self.regressor.feature_importances_)]

    def get_training_fit(self,
                         regress_limit=None,
                         output='median'):

        """
        Find out how well the training samples fit the model
        :param output: Metric to be omputed from the random forest (options: 'mean','median','sd')
        :param regress_limit: List of upper and lower regression limits for training fit prediction
        :return: None
        """
        if self.fit:
            # predict using held out samples and print to file
            pred = self.sample_predictions(self.data,
                                           regress_limit=regress_limit,
                                           output=output)

            self.training_results['rsq'] = pred['rsq'] * 100.0
            self.training_results['slope'] = pred['slope']
            self.training_results['intercept'] = pred['intercept']
            self.training_results['rmse'] = pred['rmse']
        else:
            raise ValueError("Model not initialized with samples")

    def get_adjustment_param(self,
                             clip=0.025,
                             data_limits=None,
                             output='median',
                             over_adjust=1.0):
        """
        get the model adjustment parameters based on training fit
        :param output: Metric to be computed from the random forest (options: 'mean','median','sd')
        :param clip: Ratio of samples not to be used at each tail end
        :param data_limits: tuple of (min, max) limits of output data
        :param over_adjust: Amount of over adjustment needed to adjust slope of the output data
        :return: None
        """

        if data_limits is None:
            data_limits = [self.data['labels'].min(), self.data['labels'].max()]

        regress_limit = [data_limits[0] + clip * (data_limits[1]-data_limits[0]),
                         data_limits[1] - clip * (data_limits[1]-data_limits[0])]

        self.get_training_fit(regress_limit=regress_limit,
                              output=output)

        if self.training_results['intercept'] > regress_limit[0]:
            self.adjustment['bias'] = -1.0 * (self.training_results['intercept'] / self.training_results['slope'])

        self.adjustment['gain'] = (1.0 / self.training_results['slope']) * over_adjust

        self.adjustment['lower_limit'] = data_limits[0]
        self.adjustment['upper_limit'] = data_limits[1]


class HRFRegressor(RFRegressor):

    """
    Hierarchical Random Forest Regressor.
    This class is designed to use multiple random forest regressors.
    The features in each random forest regressor must be specified.
    (based on hierarchical regression of available features)
    """

    time_it = False

    def __init__(self,
                 data=None,
                 regressor=None,
                 **kwargs):

        super(RFRegressor, self).__init__(data,
                                          regressor)

        if regressor is not None:
            if type(regressor).__name__ not in ('list', 'tuple'):
                regressor = [regressor]

        feature_list_ = list(reg.features for reg in regressor)
        feature_index_ = list(reversed(sorted(range(len(feature_list_)),
                                              key=lambda x: len(feature_list_[x]))))

        self.features = list(feature_list_[idx] for idx in feature_index_)
        self.regressor = list(regressor[idx] for idx in feature_index_)

        if data is not None:
            if type(data).__name__ not in ('list', 'tuple'):
                data = [data]
            self.data = list(data[idx] for idx in feature_index_)
        else:
            self.data = data

        self.feature_index = None

    def __repr__(self):

        if self.regressor is None:
            repr_regressor = ['<empty>']
        elif type(self.regressor).__name__ in ('list', 'tuple'):
            repr_regressor = list(regressor.__repr__() for regressor in self.regressor)
        else:
            repr_regressor = [self.regressor.__repr__()]

        return "Hierarchical regressor object" + \
            "\n---\nRegressors: \n---\n{}".format('\n'.join(repr_regressor)) + \
            "\n---\n\n"

    @Timer.timing(time_it)
    def regress_raster(self,
                       raster_obj,
                       outfile=None,
                       outdir=None,
                       band_name='prediction',
                       output_type='median',
                       array_multiplier=1.0,
                       array_additive=0.0,
                       out_data_type=gdal.GDT_Float32,
                       nodatavalue=None,
                       **kwargs):

        """Tree variance from the RF regressor
        :param raster_obj: Initialized Raster object with a 3d array
        :param outfile: name of output file
        :param array_multiplier: rescale data using this value
        :param array_additive: Rescale data using this value
        :param out_data_type: output raster data type
        :param nodatavalue: No data value for output raster
        :param band_name: Name of the output raster band
        :param outdir: output folder
        :param output_type: Should the output be standard deviation ('sd'),
                            variance ('var'), or prediction ('pred'),
                            or 'conf' for confidence interval
        :returns: Output as raster object
        """

        self.feature_index = list(list(raster_obj.bnames.index(feat) for feat in feat_grp)
                                  for feat_grp in self.features)

        return _Regressor.regress_raster(self,
                                         raster_obj,
                                         outfile=outfile,
                                         outdir=outdir,
                                         band_name=band_name,
                                         output_type=output_type,
                                         out_data_type=out_data_type,
                                         nodatavalue=nodatavalue,
                                         array_multiplier=array_multiplier,
                                         array_additive=array_additive,
                                         **kwargs)

    def predict(self,
                arr,
                output_type='median',
                **kwargs):
        """
        Calculate random forest model prediction, variance, or standard deviation.
        Variance or standard deviation is calculated across all trees.
        Tiling is necessary in this step because large numpy arrays can cause
        memory issues during creation.

        :param arr: input 2d array (axis 0: features (pixels), axis 1: bands)

        :param output_type: which output to produce,
                       choices: ['sd', 'var', 'median', 'mean', 'full']
                       where 'sd' is for standard deviation,
                       'var' is for variance
                       'median' is for median of tree outputs
                       'mean' is for mean of tree outputs
                       'full' is for the full spectrum of the leaf nodes' prediction

        :param kwargs: Keyword arguments:
                        ntile_max: Maximum number of tiles up to which the
                                   input image or array is processed without tiling (default = 5).
                                   You can choose any (small) number that suits the available memory.
                        tile_size: Number of pixels in each tile (default = 1024)
                        gain: Adjustment of the predicted output by linear adjustment of gain (slope)
                        bias: Adjustment of the predicted output by linear adjustment of bias (intercept)
                        upper_limit: Limit of maximum value of prediction
                        lower_limit: Limit of minimum value of prediction
                        intvl: Prediction interval width (default: 95 percentile)
                        uncert_dict: Dictionary specifying the indices of
                                   feature bands (keys) and their corresponding
                                   uncertainty bands (values)
                        n_rand: Number of random values to generate in the uncertainty range (default: 5)
                        half_range (Boolean): If the input and output uncertainty values are
                                   False  - full range (x +/- a), or
                                   True   - half range (x +/- a/2)

        :return: 1d image array (that will need reshaping if image output)
        """

        if output_type == 'full':
            raise ValueError('Output type "full" is not supported for this class')

        return super(HRFRegressor, self).predict(arr,
                                                 output_type=output_type,
                                                 **kwargs)

    def regress_tile(self,
                     arr,
                     tile_start=None,
                     tile_end=None,
                     output_type='median',
                     nodatavalue=None,
                     intvl=None,
                     min_variance=None,
                     **kwargs):

        """
        Method to regress each tile of the image with regressor hierarchy
        :param arr: input 2D array to process (rows = elements, columns = features)
        :param tile_start: pixel location of tile start
        :param tile_end: pixel location of tile end
        :param nodatavalue: No data value
        :param output_type: Type of output to produce,
                   choices: ['sd', 'var', 'full', 'mean', 'median']
                   where 'sd' is for standard deviation,
                   'var' is for variance
                   'full is for all leaf outputs
                   'median' is for median of tree outputs
                   'mean' is for mean of tree outputs
        :param intvl: Prediction interval width (default: 95 percentile)
        :param min_variance: Minimum variance after which to cutoff
        :param kwargs: Keyword arguments:
                ntile_max: Maximum number of tiles up to which the
                           input image or array is processed without tiling (default = 9).
                           You can choose any (small) number that suits the available memory.
                tile_size: Number of pixels in each tile (default = 1024)
                gain: Adjustment of the predicted output by linear adjustment of gain (slope)
                bias: Adjustment of the predicted output by linear adjustment of bias (intercept)
                upper_limit: Limit of maximum value of prediction
                lower_limit: Limit of minimum value of prediction
                intvl: Prediction interval width (default: 95 percentile)
                uncert_dict: Dictionary specifying the indices of
                           feature bands (keys) and their corresponding
                           uncertainty bands (values)
                n_rand: Number of random values to generate in the uncertainty range (default: 5)
                half_range (Boolean): If the input and output uncertainty values are
                           False  - full range (x +/- a), or
                           True   - half range (x +/- a/2)
        :return: numpy 1-D array
        """
        # leaf variance limit for sd or var output type
        if min_variance is None:
            if intvl is not None:
                min_variance = (1 - intvl/100.) * np.min(arr.astype(np.float32))
            else:
                min_variance = 0.05 * np.min(arr.astype(np.float32))

        # define input array shape param
        if tile_end is None:
            tile_end = arr.shape[0]
        if tile_start is None:
            tile_start = 0

        # List of index of bands to be used for regression
        if 'feature_index' in kwargs:
            feature_index = kwargs['feature_index']
        elif self.feature_index is not None:
            feature_index = self.feature_index
        else:
            feature_index = np.tile(np.array(range(0, arr.shape[0])),
                                    (len(self.regressor), 1))

        # initialize output tile
        out_tile = np.zeros([tile_end - tile_start])

        # list of list of array locations where
        # all feature bands for a regressor are available for regression
        # the self.regressor list should be in decreasing order of number of feature bands
        tile_index = list()

        for ii, _ in enumerate(self.regressor):
            # array locations where all features are available for regression
            reg_index = np.where(np.apply_along_axis(lambda x: np.all(x[feature_index[ii]] != nodatavalue),
                                                     1,
                                                     arr[tile_start:tile_end, :]))[0]

            if len(tile_index) == 0:
                tile_index.append(reg_index)
            else:
                # check to see if the indices found earlier for this regressor are
                # already available for previous regressor. if so, mask them out
                for index_list in tile_index:
                    intersecting_index = np.where(np.in1d(reg_index, index_list))[0]

                    mask = np.zeros(reg_index.shape,
                                    dtype=bool) + True
                    mask[intersecting_index] = False

                    reg_index = reg_index[np.where(mask)[0]]

                # add array indices/locations not used by previous regressor to list
                tile_index.append(reg_index)

        for ii, regressor in enumerate(self.regressor):
            Opt.cprint(' . {}'.format(str(ii + 1)), newline='')

            temp_tile = np.zeros([tile_index[ii].shape[0]]) * 0.0

            if temp_tile.shape[0] > 0:

                temp_arr = (arr[tile_index[ii][:, np.newaxis] + tile_start, feature_index[ii]] *
                            kwargs['band_multipliers'][feature_index[ii]]) + \
                    kwargs['band_additives'][feature_index[ii]]

                # initialize output array for this regressor
                tile_arr = np.zeros([regressor.trees, tile_index[ii].shape[0]], dtype=float)

                if output_type in ('mean', 'median'):

                    # calculate tree predictions for each pixel in the input array
                    for jj, tree_ in enumerate(regressor.regressor.estimators_):
                        tile_arr[jj, :] = tree_.predict(temp_arr)

                    if output_type == 'median':
                        temp_tile = np.median(tile_arr, axis=0)
                    elif output_type == 'mean':
                        temp_tile = np.mean(tile_arr, axis=0)

                elif output_type in ('sd', 'var'):

                    # calculate variance of output across all the leaf nodes
                    for jj, tree_ in enumerate(regressor.regressor.estimators_):
                        tile_arr[jj, :] = tree_.predict(temp_arr)

                        var_tree = tree_.tree_.impurity[tree_.apply(temp_arr)]

                        var_tree[var_tree < min_variance] = min_variance
                        mean_tree = tree_.predict(temp_arr)
                        temp_tile += var_tree + mean_tree ** 2

                    predictions = np.mean(tile_arr, axis=0)

                    temp_tile /= len(regressor.regressor.estimators_)
                    temp_tile -= predictions ** 2.0
                    temp_tile[temp_tile < 0.0] = 0.0

                    if output_type == 'sd':
                        temp_tile = temp_tile ** 0.5
                else:
                    raise RuntimeError("Unsupported output type or no output type specified")

                if len(regressor.adjustment) > 0:

                    if 'gain' in regressor.adjustment:
                        temp_tile = temp_tile * regressor.adjustment['gain']

                    if output_type not in ('sd', 'var'):

                        if 'bias' in regressor.adjustment:
                            temp_tile = temp_tile + regressor.adjustment['bias']

                        if 'upper_limit' in regressor.adjustment:
                            temp_tile[temp_tile > regressor.adjustment['upper_limit']] = \
                                regressor.adjustment['upper_limit']

                        if 'lower_limit' in regressor.adjustment:
                            temp_tile[temp_tile < regressor.adjustment['lower_limit']] = \
                                regressor.adjustment['lower_limit']

                # write output to the output tile
                out_tile[tile_index[ii]] = temp_tile

        return out_tile
