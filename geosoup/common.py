from decimal import Decimal, getcontext
from itertools import takewhile, repeat
from functools import wraps
import numpy as np
import datetime
import logging
import fnmatch
import random
import psutil
import shutil
import ftplib
import time
import copy
import gzip
import sys
import os


__all__ = ['Sublist',
           'Handler',
           'FTPHandler',
           'Logger',
           'Timer',
           'Opt']

julian_days = {
                  1: (1, 31),
                  2: (32, 59),
                  3: (60, 90),
                  4: (91, 120),
                  5: (121, 151),
                  6: (152, 181),
                  7: (182, 212),
                  8: (213, 243),
                  9: (244, 273),
                  10: (274, 304),
                  11: (305, 334),
                  12: (335, 365)
                }


class Sublist(list):
    """
    Class to handle list operations
    """
    def __eq__(self,
               other):
        """
        Check for a = other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] == other)

        if len(temp) == 1:
            temp = temp[0]

        return temp

    def __gt__(self,
               other):
        """
        Check for a > other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] > other)

        if len(temp) == 1:
            temp = temp[0]

        return temp

    def __ge__(self,
               other):
        """
        Check for a >= other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] >= other)

        if len(temp) == 1:
            temp = temp[0]

        return temp

    def __lt__(self,
               other):
        """
        Check for a < other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] < other)

        if len(temp) == 1:
            temp = temp[0]

        return temp

    def __le__(self,
               other):
        """
        Check for a <= other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] <= other)

        if len(temp) == 1:
            temp = temp[0]

        return temp

    def __ne__(self,
               other):
        """
        Check for a != other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] != other)

        if len(temp) == 1:
            temp = temp[0]

        return temp

    def __add__(self,
                other):
        """
        Adding another list
        :param other: other list
        :return: list
        """
        return Sublist(list(self) + list(other))

    def __getitem__(self,
                    item):
        """
        Method to get item(s) from the list using list or number as index
        :param item: Number or list of numbers
        :return: List
        """

        try:
            if isinstance(item, list):
                return list(list(self)[i] for i in item)
            else:
                return list(self)[item]

        except (TypeError, KeyError):
            print("List index not a number or list of numbers")

    def flatten(self):
        """
        Flatten a Sublist object
        returns: 1D Sublist object
        """
        if len(self) == 0:
            return self
        if isinstance(self, list):
            return Sublist(self[0]).flatten() + Sublist(self[1:]).flatten()
        return self[:1] + Sublist(self[1:]).flatten()

    def range(self,
              llim,
              ulim,
              index=False):
        """
        Make a subset of the list by only including elements between ulim and llim.
        :param ulim: upper limit
        :param llim: lower limit
        :param index: If the function should return index of values that lie within the limits
        :return: List
        """
        if index:
            return list(i for i, x in enumerate(self) if llim <= x <= ulim)
        else:
            return list(x for x in self if llim <= x <= ulim)

    def add(self,
            elem):
        """
        Add two lists
        :param elem: Another list or element
        :return: list
        """
        if isinstance(elem, list):
            for val in elem:
                self.append(val)
            return self
        else:
            self.append(elem)
            return self

    def remove_by_loc(self,
                      elem):
        """
        Method to remove a Sublist element with index 'other'
        :param elem: Index or list of indices
        :return: Sublist
        """
        mask = np.zeros(len(self), dtype=bool) + True
        mask[set(list(elem))] = False
        return (np.array(self)[np.where(mask)]).tolist()

    def remove(self,
               elem):
        """
        Method to remove list item or sublist in a 1d int or float list
        :param elem: item or list
        :return: list
        """
        if type(elem).__name__ not in ('list', 'tuple', 'generator'):
            return (np.array(self)[~np.in1d(np.array(self), np.array(elem))]).tolist()

        else:
            elem_ = list(set(list(elem)))
            return (np.array(self)[~np.in1d(np.array(self), np.array(elem_))]).tolist()

    @staticmethod
    def list_size(query_list):
        """
        Find size of a list object even if it is a one element non-list
        :param query_list: List to be sized

        """

        if isinstance(query_list, list):
            return len(query_list)
        else:
            return 1

    def remove_by_percent(self,
                          percent):
        """
        Method to remove randomly selected elements by percentage in a list
        :param percent: Percentage (Range 0-100)
        :return: list
        """
        nelem = len(self)
        nelem_by_percent = int(round((float(nelem)*float(100 - percent))/float(100)))
        return random.sample(self, nelem_by_percent)

    def random_selection(self,
                         num=1,
                         systematic=False):

        """
        Method to select a smaller number of samples from the Samples object
        :param num: Number of samples to select
        :param systematic: If the numbers returned should be systematic
                           Used only for large number of samples
        :return: Samples object
        """

        arr = np.array(self)

        if num >= len(self):
            return self

        elif systematic:
            diff_seq = self.custom_list(0, len(self)-1, step=int(float(len(self))/float(num)))
            temp = arr[np.array(diff_seq)]
            return Sublist(temp)

        else:
            return Sublist(np.random.choice(arr, size=num, replace=False))

    def tuple_by_pairs(self):
        """
        Make a list of tuple pairs of consequetive list elements
        :return: List of tuples
        """

        return Sublist((self[i], self[i+1]) for i in range(len(self) - 1))

    @classmethod
    def custom_list(cls,
                    start,
                    end,
                    step=None):
        """
        List with custom first number but rest all follow same increment; integers only
        :param start: starting integer
        :param end: ending integer
        :param step: step integer
        :return: list of integers
        """

        # end of list
        end = end + step - (end % step)

        # initialize the list
        out = Sublist()

        # make iterator
        if step is not None:
            if start % step > 0:
                out.append(start)
                start = step * (start < step) + (start + step - (start % step)) * (start > step)
            iterator = range(start, end, step)
        else:
            step = 1
            iterator = range(start, step, end)

        # add the rest of the list to out
        for i in iterator:
            out.append(i)

        return out

    def sublistfinder(self,
                      pattern):
        """
        Find the location of sub-list in a python list, no repetitions in mylist or pattern
        :param pattern: shorter list
        :return: list of locations of elements in mylist ordered as in pattern
        """
        return Sublist(self.index(x) if x in self else -1 for x in pattern)

    @staticmethod
    def hist(var_list,
             nbins=20,
             minmax=None):

        """
        Method to generate a histogram from a list of numbers
        :param nbins: Number of bins to group the numbers in a list
        :param minmax: Tuple of minimum and maximum values to be used
                       as criteria of inclusion
        :returns: frequency list (list of elements in each bin), list of bin edges

        """

        if minmax is None:
            minmax = (min(var_list), max(var_list))

        bin_edges = Sublist.frange(minmax[0], minmax[1], div=nbins)
        freq_list = list(0 for _ in range(len(bin_edges) - 1))

        for elem in var_list:
            for ii in range(len(bin_edges) - 1):
                if bin_edges[ii] <= elem < bin_edges[ii + 1]:
                    freq_list[ii] += 1
                  
        return freq_list, bin_edges

    @staticmethod
    def hist_equalize(list_dicts,
                      num=None,
                      pctl=50,
                      nbins=20,
                      var=None,
                      minmax=None):

        """
        Method to clip the frequency bins at a threshold and randomly select the
        elements from the remaining elements inside the bin
        :param list_dicts: List of dictionaries with one of key as the specified variable
        :param var: Name of the key, variable of interest
        :param pctl: Bin clipping percentile, ignored if num is specified
        :param num: Bin clipping value
        :param nbins: Number of bins
        :param minmax: Tuple of minimum and maximum values to be used
                       as criteria of inclusion
        :returns: list of dictionaries with clipped bins based on num or pctl
        """

        if var is None:
            raise ValueError('Variable not specified')
        else:
            var_list = np.array(list(elem[var] for elem in list_dicts))

            if minmax is None:
                minmax = (np.min(var_list),
                          np.max(var_list))

            freq_list, bin_edges = np.histogram(var_list,
                                                bins=nbins,
                                                range=minmax)

            if num is None:
                pctl_val = Sublist.percentile(freq_list,
                                              pctl=pctl)
            else:
                pctl_val = num

            out_list = list()

            for ii in range(len(freq_list)):

                if ii == (len(freq_list) - 1):
                    indices = np.where((var_list >= bin_edges[ii]) &
                                       (var_list <= bin_edges[ii + 1]))[0]

                else:
                    indices = np.where((var_list >= bin_edges[ii]) &
                                       (var_list < bin_edges[ii + 1]))[0]

                if indices.shape[0] > pctl_val:
                    out_indices = np.random.choice(indices,
                                                   size=pctl_val,
                                                   replace=False)
                else:
                    out_indices = indices

                out_list += list(list_dicts[jj] for jj in out_indices.tolist())

            return out_list

    @staticmethod
    def calc_parabola_param(pt1, pt2, pt3):
        """
        define a parabola using three points
        :param pt1: First point (x,y)
        :param pt2: Second point (x,y)
        :param pt3: Third point (x,y)
        :return tuple of a, b, and c for parabola a(x^2) + b*x + c = 0
        """
        x1, y1 = pt1
        x2, y2 = pt2
        x3, y3 = pt3

        _m_ = (x1 - x2) * (x1 - x3) * (x2 - x3)
        a_param = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / _m_
        b_param = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / _m_
        c_param = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / _m_

        return a_param, b_param, c_param

    @staticmethod
    def moving_average(arr,
                       n=3,
                       cascaded=False):
        """
        Method to smooth an array of numbers using moving array method
        :param arr: Input array
        :param n: Number of elements to consider for moving average. Must be an odd number (default: 3)
        :param cascaded: If the smoothing should be cascaded from the specified moving average to the lowest(1)
        :return: smoothed array
        """
        if type(arr) in (list, tuple, dict, set):
            arr_copy = np.array(list(copy.deepcopy(arr)))
        elif type(arr) == np.ndarray:
            arr_copy = arr.copy()
        else:
            raise ValueError("Input array type not understood")

        dtype = arr_copy.dtype

        if n < 3:
            raise ValueError('n cannot be less than 3')

        if cascaded:
            ker_list = list(ker_size for ker_size in reversed(range(n + 1)) if ker_size % 2 == 1)
            if ker_list[-1] == 0:
                ker_list = ker_list[:-1]
        else:
            if n % 2 == 0:
                raise ValueError('n cannot be an even number')
            ker_list = [n]

        for ker_size in ker_list:
            if ker_size > 1:
                tail = int((ker_size - 1) / 2)

                ret = np.cumsum(np.concatenate([arr_copy[0:tail],
                                                arr_copy,
                                                arr_copy[-tail:]]),
                                dtype=np.float32)

                ret[ker_size:] = ret[ker_size:] - ret[:-ker_size]

                arr_copy = ret[ker_size - 1:] / ker_size

        arr_copy = arr_copy.astype(dtype)

        if type(arr) in (list, tuple, dict, set):
            return arr_copy.tolist()
        else:
            return arr_copy

    def count_in_range(self,
                       llim,
                       ulim):
        """
        Find elements in a range
        :param ulim: upper limit
        :param llim: lower limit
        :return: count, list
        """
        return len([i for i in self if llim <= i < ulim])

    @classmethod
    def frange(cls,
               start,
               end,
               step=None,
               div=None):
        """
        To make list from float arguments
        :param start: start number
        :param end: end number
        :param step: step
        :param div: Number of divisions between start and end
        :return: list
        """
        if end > start:
            return_rev = False
        elif end < start:
            start, end = end, start
            return_rev = True
        else:
            raise ValueError("Start and end value are the same!")

        if div is not None:
            temp = Sublist(start + i * ((end-start)/float(div)) for i in range(0, div)) + [end]

        elif step is not None:
            if (end - start) % step == 0.0:
                temp = Sublist(start + i * step for i in range(0, int((end - start) / step))) + [end]
            else:
                temp = Sublist(start + i * step for i in range(0, int((end - start) / step) + 1)) + [end]

        else:
            raise ValueError("No step or division defined")

        if return_rev:
            return Sublist(reversed(temp))
        else:
            return temp

    def reverse(self):
        """reversed list"""
        return Sublist(reversed(self))

    def shuffle(self):
        """
        shuffle the list items
        :return: list
        """
        x_ = self.copy()
        np.random.shuffle(x_)
        return x_

    def copy(self):
        """returns copied instance"""
        return copy.deepcopy(self)

    @classmethod
    def column(cls,
               matrix,
               i):
        """
        Get column of a numpy matrix
        :param matrix: Numpy matrix
        :param i: index
        :return: List
        """
        mat = matrix[:, i].tolist()
        return Sublist(elem[0] for elem in mat)

    @classmethod
    def row(cls,
            matrix,
            i):
        """
        Get row of a numpy matrix
        :param matrix: Numpy matrix
        :param i: index
        :return: List
        """
        mat = matrix[i].tolist()
        return Sublist(mat[0])

    def mean(self):
        """
        calculate mean of an array
        :return: float
        """
        return float(sum(self)) / max(len(self), 1)

    def max(self):
        """
        Calculate max of the list
        :return:
        """
        if type(max(self)).__name__ == 'list':
            return max(self)[0]
        else:
            return max(self)

    def median(self):
        """
        calculate median of an array
        :return: float
        """
        return self.percentile(self, pctl=50)

    @staticmethod
    def percentile(arr,
                   pctl=95.0):
        """
        Method to output percentile in an array
        :param arr: input numpy array or iterable
        :param pctl: percentile value
        :return:
        """

        return np.percentile(arr, pctl, interpolation='nearest')

    @staticmethod
    def pctl_interval(arr, intvl=95.0):
        """
        Method to calculate the width of a percentile interval
        :param arr: input numpy array or iterable
        :param intvl: Interval to calculate (default: 95th percentile)
        :return: scalar
        """
        lower = Sublist.percentile(arr, (100.0 - intvl)/2.0)
        upper = Sublist.percentile(arr, 100.0 - (100.0 - intvl)/2.0)

        return np.abs((upper - lower)/2.0)

    @staticmethod
    def std_dev(arr):
        """
        Method to calculate standard deviation of an array
        :param arr: input array
        :return: standard deviation
        """
        return np.std(arr)

    @staticmethod
    def reduce(array,
               method='mean',
               axis=0):
        """
        Method to reduce a 2D or 3D array using a specific method
        :param array: Numpy array
        :param method: Method to use for reduction (options: mean, median, std_dev,
                                                        pctl_xx, min, max, default:mean
                                                             here xx is the percentile)
        :param axis: Axis to apply the reducer along
        :return: float
        """

        if (method not in ('mean', 'median', 'std_dev', 'min', 'max')) or ('pctl' not in method):
            raise ValueError("Reducer = {} is not implemented".format(method))

        return (method == 'mean') * np.mean(array, axis=axis) + \
               (method == 'median') * np.median(array, axis=axis) + \
               (method == 'std_dev') * np.std(array, axis=axis) + \
               (method == 'min') * np.min(array, axis=axis) + \
               (method == 'max') * np.max(array, axis=axis) + \
               ('pctl' in method) * np.percentile(array, int(method.split('_')[1]), axis=axis)


class Handler(object):
    """
    Class to handle file and folder operations
    """
    def __init__(self,
                 filename=None,
                 basename=None,
                 dirname=None):

        self.filename = filename

        try:
            self.basename = os.path.basename(filename)
        except:
            self.basename = basename

        try:
            self.dirname = os.path.dirname(filename)
        except:
            self.dirname = dirname

        self.sep = os.path.sep

    def __repr__(self):
        if self.filename is not None:
            return '<Handler for {}>'.format(self.filename)
        elif self.dirname is not None:
            return '<Handler for {}>'.format(self.dirname)
        elif self.basename is not None:
            return '<Handler for {}>'.format(self.basename)
        else:
            return '<Handler {s}____{s}>'.format(s=self.sep)

    def dir_create(self):
        """
        Create dir if it doesnt exist
        :param: directory to be created
        """
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

    def add_to_filename(self,
                        string=None,
                        timestamp=False):
        """
        Method to append a string to a filename, before the .xxx extension
        :param string: String to append
        :param timestamp: If timestamp (upto seconds) should be added to a filename, before the .xxx extension
                          This can be used in addition to the string
        :return:
        """

        if timestamp:
            timestamp = datetime.datetime.now().isoformat().replace('-', '').replace(':', '').split('.')[0]
        else:
            timestamp = ''

        if string is not None:
            string += timestamp
        else:
            string = timestamp

        file_name, file_ext = os.path.splitext(self.basename)

        file_name += string

        self.basename = file_name + file_ext

        if self.dirname is not None:
            return self.dirname + self.sep + self.basename
        else:
            return self.basename

    def file_rename_check(self):
        """
        Change the file name if already exists by incrementing trailing number
        :return filename
        """
        if not os.path.isfile(self.filename):
            self.dir_create()

        # if file exists then rename the input filename
        counter = 1
        while os.path.isfile(self.filename):
            components = self.basename.split('.')
            if len(components) < 2:
                self.filename = self.dirname + os.path.sep + self.basename + '_' + str(counter)
            else:
                self.filename = self.dirname + os.path.sep + ''.join(components[0:-1]) + \
                           '_(' + str(counter) + ').' + components[-1]
            counter = counter + 1
        return self.filename

    def file_remove_check(self):
        """
        Remove a file silently; if not able to remove, change the filename and move on
        :return filename
        """

        # if file does not exist, try to create dir
        if not os.path.isfile(self.filename):
            self.dir_create()

        # if file exists then try to delete or
        # get a filename that does not exist at that location
        counter = 1
        while os.path.isfile(self.filename):
            # print('File exists: ' + filename)
            # print('Deleting file...')
            try:
                os.remove(self.filename)
            except Exception as e:
                print(e)
                components = self.basename.split('.')
                if len(components) < 2:
                    self.filename = self.dirname + os.path.sep + self.basename + '_' + str(counter)
                else:
                    self.filename = self.dirname + os.path.sep + ''.join(components[0:-1]) + \
                               '_(' + str(counter) + ').' + components[-1]
                # print('Unable to delete, using: ' + filename)
                counter = counter + 1
        return self.filename

    def file_exists(self):
        """
        Check file existence
        :return: Bool
        """
        return os.path.isfile(self.filename)

    def dir_exists(self):
        """
        Check folder existence
        :return: Bool
        """
        return os.path.isdir(self.dirname)

    def file_delete(self):
        """
        Delete a file
        """
        if self.file_exists():
            os.remove(self.filename)

    def copy_file(self,
                  other_dir,
                  replace=True,
                  move=False):
        """
        Copy a file from one location to another
        :param other_dir: Name of output directory
        :param replace: If the existing output file should be replaced?
        :param move: If the file should be moved instead of copy
        """
        if other_dir[-1] == '/':
            out_filename = other_dir + self.basename
        else:
            out_filename = other_dir + self.sep + self.basename

        if replace:
            out_filename = Handler(out_filename).file_remove_check()

        shutil.copy(self.filename,
                    out_filename)

        if move:
            self.file_delete()

    def copy_dir(self,
                 other_dir):

        """
        Method to copy a directory to another
        :param other_dir: Name of output directory
        """
        shutil.copytree(self.dirname,
                        other_dir)

    def file_lines(self,
                   nlines=True,
                   bufsize=102400):
        """
        Find number of lines or get text lines in a text or csv file
        :param nlines: If only the number of lines in a file should be returned
        :param bufsize: size of buffer to be read
        :return: list or number
        """
        with open(self.filename, 'r') as f:
            bufgen = takewhile(lambda x: x, (f.read(bufsize) for _ in repeat(None)))

            if nlines:
                val = sum(buf.count('\n') for buf in bufgen if buf)
            else:
                val = list()
                remaining = ''
                for buf in bufgen:
                    if buf:
                        temp_lines = (remaining + buf).split('\n')
                        if len(temp_lines) <= 1:
                            remaining += ''.join(temp_lines)
                        else:
                            val += temp_lines[:-1]
                            remaining = temp_lines[-1]
        return val

    def read_csv_as_array(self):
        """
        Read csv file as numpy array
        """
        csv_data = np.genfromtxt(self.filename,
                                 delimiter=',',
                                 names=True,
                                 encoding="utf-8",
                                 dtype=None)

        names = [csv_data.dtype.names[i] for i in range(0, len(csv_data.dtype.names))]
        return names, csv_data

    def read_text_by_line(self):
        """
        Read text file line by line and output a list of text lines
        """
        with open(self.filename) as f:
            content = f.readlines()
        return [x.strip() for x in content]

    def read_array_from_csv(self,
                            delim=",",
                            array_1d=False,
                            nodataval=None):
        """
        Read array from file and output a numpy array. There should be no header.
        :param delim: Delimiter (default ',')
        :param array_1d: Should the array be reshaped to 1-dimensional array? (default: False)
        :param nodataval: Data values to be removed from 1-dimensional array
                          (ignored if array_1d flag is false)
        :returns: Numpy array (2d or 1d)
        """
        with open(self.filename) as f:
            content = f.readlines()
            lines = [x.strip() for x in content]
            ncols = len(lines[0].split(delim))
            nrows = len(lines)
            arr = np.zeros((nrows, ncols))
            for i, line in enumerate(lines):
                arr[i:] = [float(elem.strip()) for elem in line.split(delim)]

        if array_1d:
            lenr, lenc = arr.shape
            arr_out = arr.reshape(lenr * lenc)
            if nodataval is not None:
                arr_out = arr_out[arr_out != nodataval]
            return arr_out
        else:
            return arr

    def extract_gz(self,
                   dirname=None,
                   add_ext=None):
        """
        Extract landsat file to a temp folder
        :param dirname: Folder to extract all files to
        :param add_ext: Extension to be added to extracted file
        """

        if add_ext is None:
            add_ext = ''
        else:
            add_ext = '.' + str(add_ext)

        # define a temp file with scene ID
        if dirname is not None:
            tempfile = dirname + self.sep + self.basename.split('.gz')[0] + add_ext
        else:
            tempfile = self.dirname + self.sep + self.basename.split('.gz')[0] + add_ext

        # remove if file already exists
        temp = Handler(tempfile)
        temp.file_remove_check()

        # extract infile to temp folder
        if self.filename.endswith(".gz"):
            Opt.cprint('Extracting {} to {}'.format(self.basename,
                                                    temp.basename))
            with gzip.open(self.filename, 'rb') as gf:
                file_content = gf.read()
                with open(tempfile, 'wb') as fw:
                    fw.write(file_content)

        else:  # not a tar.gz archive
            raise TypeError("Not a .gz archive")

    def get_size(self,
                 unit='kb',
                 precision=3,
                 as_long=True):
        """
        Function to get file size
        :param unit: Unit to get file size in (options: 'b', 'kb', 'mb', 'gb', 'tb', 'pb', 'bit')
        :param precision: Precision to report
        :param as_long: Should the output be returned as a long integer? This truncates any decimal value
        :return: long integer
        """
        size = os.path.getsize(self.filename)

        getcontext().prec = precision

        if unit not in ('b', 'kb', 'mb', 'gb', 'tb', 'pb', 'bit'):
            raise NotImplementedError('Unrecognized size unit')

        output = (unit == 'bit') * float(Decimal(size)*Decimal(2**3)) + \
                 (unit == 'b') * float(Decimal(size)) + \
                 (unit == 'kb') * float(Decimal(size)/Decimal(2**10)) + \
                 (unit == 'mb') * float(Decimal(size)/(Decimal(2**20))) + \
                 (unit == 'gb') * float(Decimal(size)/(Decimal(2**30))) + \
                 (unit == 'tb') * float(Decimal(size)/(Decimal(2**40))) + \
                 (unit == 'pb') * float(Decimal(size)/(Decimal(2**50)))

        output = round(output, precision)

        if as_long:
            output = int(output)

        return output

    def write_list_to_file(self,
                           input_list,
                           rownames=None,
                           colnames=None,
                           delim=", ",
                           append=False):
        """
        Function to write list to file
        with each list item as one line
        :param input_list: input text list
        :param rownames: list of row names strings
        :param colnames: list of column name strings
        :param delim: delimiter (default: ", ")
        :param append: if the lines should be appended to the file
        :return: write to file
        """
        if len(input_list) == 0:
            raise ValueError('Empty input list supplied')

        elif type(input_list[0]).__name__ in ('list', 'tuple'):
            if rownames is not None:
                if len(rownames) != len(input_list):
                    raise ValueError('Row name list does not have sufficient elements')
                else:
                    input_list = list([rownames[i]] + elem for i, elem in enumerate(input_list))

            if colnames is not None:
                if len(rownames) != len(input_list[0]):
                    raise ValueError('Column name list does not have sufficient elements')
                else:
                    input_list = [colnames] + input_list

            input_list = '\n'.join(list(delim.join(list(str(elem) for elem in line)) for line in input_list))

        elif type(input_list[0]).__name__ == 'str':
            input_list = '\n'.join(input_list)

        else:
            raise ValueError('Input list not in types: list of list, list of tuples, list of strings')

        # create dir path if it does not exist
        self.dir_create()

        # write to file
        if append:
            open_type = 'a'  # append
        else:
            open_type = 'w'  # write

        with open(self.filename, open_type) as fileptr:
            fileptr.write('{}\n'.format(input_list))

    def write_slurm_script(self,
                           job_name='pyscript',
                           time_in_mins=60,
                           cpus=1,
                           ntasks=1,
                           mem=2000,
                           array=False,
                           iterations=1,
                           **kwargs):
        """
        Write slurm script for the given parameters
        :param job_name: Name of the SLURM job
        :param time_in_mins: expected run time in minutes
        :param cpus: Number of CPUs requested
        :param ntasks: Number of tasks
        :param mem: Memory requested (in MB)
        :param array: (bool) If using job arrays
        :param iterations: Job array upper limit (e.g. 132 for 1-132 array)
        :param kwargs: key word arguments
        :return:
        """
        script_dict = {
            'bash': '#!/bin/bash',
            'job-name': '#SBATCH --job-name=' + job_name,
            'time': '#SBATCH --time=' + str(datetime.timedelta(minutes=time_in_mins)),
            'cpus': '#SBATCH --cpus-per-task=' + str(cpus),
            'ntasks': '#SBATCH --ntasks=' + str(ntasks),
            'mem': '#SBATCH --mem=' + str(mem),
            'partition': '#SBATCH --partition=all',
            'array_def': '#SBATCH --array=1-' + str(iterations),
            'array_out': '#SBATCH --output=/scratch/rm885/support/out/slurm-jobs/iter_%A_%a.out',
            'out': '#SBATCH --output=/scratch/rm885/support/out/slurm-jobs/slurm_%j.out',
            'date': 'date',
            'array_echo': 'echo "Job ID is"$SLURM_ARRAY_TASK_ID'
        }

        script_list = [script_dict['bash'],
                       script_dict['job-name'],
                       script_dict['time'],
                       script_dict['cpus'],
                       script_dict['mem'],
                       script_dict['ntasks']]

        if array:
            script_list += [script_dict['array_def'],
                            script_dict['array_out'],
                            script_dict['array_echo']]

        script_list.append(script_dict['date'])

        for key, value in kwargs.items():
            script_list.append(value)

        script_list.append(script_dict['date'])

        self.write_list_to_file(script_list)

    def write_numpy_array_to_file(self,
                                  np_array,
                                  colnames=None,
                                  rownames=None,
                                  delim=","):
        """
        Write numpy array to file
        :param np_array: Numpy 2d array to be written to file
        :param colnames: list of column name strings
        :param rownames: list of row name strings (should have the same length as np_array axis 0
        :param delim: Delimiter (default: ", ")
        """
        if colnames is not None:
            header = delim.join(colnames)
        else:
            header = None

        if len(np_array.shape) == 1:
            np_array = np_array[:, np.newaxis]

        if rownames is not None:
            np_array = np.hstack(np.array(rownames)[:, np.newaxis], np_array)
            if header is not None:
                header = delim + header

        np_list = []
        np_list += np_array.tolist()
        np_list = list(delim.join(list(str(elem) for elem in row_list))
                       for row_list in np_list)
        if header is not None:
            np_list = [header] + np_list

        with open(self.filename, 'w') as fileptr:
            for line in np_list:
                fileptr.write('{}\n'.format(str(line)))

    def read_from_csv(self,
                      return_dicts=False,
                      line_limit=None,
                      read_random=False,
                      percent_random=50.0,
                      verbose=False,
                      systematic=False):
        """
        Read csv as list of lists with header.
        Each row is a list and a sample point.
        :param return_dicts: If list of dictionaries should be returned (default: False)
        :param line_limit: Limits on the number of lines returned (default: None)
        :param read_random: If lines to be read should be randomly selected
        :param percent_random: What percentage of lines should be randomly selected (default: 50%)
        :param verbose: Display steps
        :param systematic: If the samples should be systematic instead of random (used only if read_random is True)
        :returns: dictionary of data
        """
        lines = list()
        index_list = list()

        n_lines = self.file_lines(nlines=True)
        if verbose:
            sys.stdout.write('Lines in file: {}\n'.format(str(n_lines)))

        if read_random:
            if 0.0 < percent_random <= 100.0:
                if verbose:
                    sys.stdout.write('Randomizing index ... \n')
                n_rand_lines = int((float(percent_random) / 100.0) * float(n_lines))
                index_list = sorted([0] + Sublist(range(1, n_lines)).random_selection(num=n_rand_lines,
                                                                                      systematic=systematic))
        if verbose:
            sys.stdout.write('Reading file : ')

        counter = 0
        perc_ = 0
        with open(self.filename, 'r') as f:
            if line_limit:
                if len(index_list) > 0:
                    if 0 < len(index_list) <= line_limit:
                        pass
                    elif len(index_list) > line_limit:
                        index_list = index_list[:(line_limit + 1)]

                    for i, line in enumerate(f):
                        if counter > line_limit:
                            break
                        if i == index_list[counter]:
                            lines.append(list(elem.strip() for elem in line.split(',')))
                            counter += 1

                            if counter > int((float(perc_) / 100.0) * float(len(index_list))):
                                if verbose:
                                    sys.stdout.write('{}..'.format(str(perc_)))
                                perc_ += 10
                    if verbose:
                        sys.stdout.write('!\n')

                else:
                    for line in f:
                        if counter > line_limit:
                            break
                        lines.append(list(elem.strip() for elem in line.split(',')))
                        counter += 1

                        if counter > int((float(perc_) / 100.0) * float(line_limit)):
                            if verbose:
                                sys.stdout.write('{}..'.format(str(perc_)))
                            perc_ += 10
                    if verbose:
                        sys.stdout.write('!\n')

            else:
                if len(index_list) > 0:

                    for i, line in enumerate(f):
                        if counter >= len(index_list):
                            break

                        if index_list[counter] == i:
                            lines.append(list(elem.strip() for elem in line.split(',')))
                            counter += 1

                            if counter > int((float(perc_) / 100.0) * float(len(index_list))):
                                if verbose:
                                    sys.stdout.write('{}..'.format(str(perc_)))
                                perc_ += 10
                    if verbose:
                        sys.stdout.write('!\n')

                else:
                    for line in f:
                        lines.append(list(elem.strip() for elem in line.split(',')))
                        counter += 1

                        if counter > int((float(perc_) / 100.0) * float(n_lines)):
                            if verbose:
                                sys.stdout.write('{}..'.format(str(perc_)))
                            perc_ += 10
                    if verbose:
                        sys.stdout.write('!\n')

        # convert col names to list of strings
        names = list(elem.strip() for elem in lines[0])

        if len(lines) > 0:
            # convert to list
            if return_dicts:
                if verbose:
                    sys.stdout.write('Converting to Dictionaries...\n')
                return list(dict(zip(names, list(self.string_to_type(elem) for elem in feat)))
                            for feat in lines[1:])
            else:
                return {
                    'feature': list(list(self.string_to_type(elem) for elem in feat)
                                    for feat in lines[1:]),
                    'name': names,
                }
        else:
            return {
                'feature': list(),
                'name': list(),
            }

    @staticmethod
    def write_to_csv(list_of_dicts,
                     outfile=None,
                     delimiter=',',
                     append=False,
                     header=True):

        if outfile is None:
            raise ValueError("No file name for writing")

        if type(list_of_dicts).__name__ not in ('tuple', 'list'):
            list_of_dicts = [list_of_dicts]

        lines = list()
        if header:
            lines.append(delimiter.join(list(list_of_dicts[0])))
        for data_dict in list_of_dicts:
            lines.append(delimiter.join(list(str(val) for _, val in data_dict.items())))

        if append:
            if os.path.isfile(outfile):
                with open(outfile, 'a') as f:
                    for line in lines:
                        f.write(line + '\n')
            else:
                with open(outfile, 'w') as f:
                    for line in lines:
                        f.write(line + '\n')
        else:
            with open(outfile, 'w') as f:
                for line in lines:
                    f.write(line + '\n')

    def find_all(self,
                 pattern='*',
                 find_dirs=False,
                 verbose=False):
        """
        Find all the names that match pattern
        :param pattern: pattern to look for in the folder
        :param find_dirs: If folder names should be searched instead of files
        :param verbose: Should the files be displayed
        """
        result = []
        # search for a given pattern in a folder path
        if pattern == '*':
            search_str = '*'
        else:
            search_str = '*' + pattern + '*'

        if self.dirname[-1] != self.sep:
            self.dirname += self.sep

        for root, dirs, files in os.walk(self.dirname):
            if find_dirs:
                for dirname in dirs:
                    if verbose:
                        Opt.cprint(os.path.join(root, dirname))

                    if fnmatch.fnmatch(dirname, search_str):
                        if str(root) in str(self.dirname) or str(self.dirname) in str(root):
                            result.append(os.path.join(root, dirname))

            else:
                for name in files:
                    if verbose:
                        Opt.cprint(os.path.join(root, name))

                    if fnmatch.fnmatch(name, search_str):
                        if str(root) in str(self.dirname) or str(self.dirname) in str(root):
                            result.append(os.path.join(root, name))
        if verbose:
            Opt.cprint('======================\nFound {} results for {} in {}:'.format(str(len(result)),
                                                                                       pattern,
                                                                                       self.dirname))
            for filename in result:
                Opt.cprint(filename)
            Opt.cprint('======================')
        return result  # list

    def find_files(self,
                   pattern='*'):
        return list(f_ for f_ in list(f for f in os.listdir(self.dirname)
                                      if os.path.isfile(os.path.join(self.dirname, f)))
                    if pattern in f_)

    @staticmethod
    def string_to_type(x):
        """
        Method to return name of the data type
        :param x: input item
        :return: string
        """
        if type(x).__name__ == 'str':
            x = x.strip()
            try:
                val = int(x)
            except ValueError:
                try:
                    val = float(x)
                except ValueError:
                    try:
                        val = str(x)
                    except:
                        val = None
            x = val
        return x


class Opt:
    """
    Class for optional miscellaneous methods
    """
    def __init__(self,
                 obj=None):
        """
        Instantiate Opt class
        """
        self.obj = obj  # mutable object

    def __repr__(self):
        return "<Optional notice class>"

    @staticmethod
    def print_memory_usage():
        """
        Function to print memory usage of the python process
        :return print to console/output
        """
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss

        if 2**10 <= mem < 2**20:
            div = float(2**10)
            suff = ' KB'
        elif 2**20 <= mem < 2**30:
            div = float(2**20)
            suff = ' MB'
        elif 2**30 <= mem < 2**40:
            div = float(2**30)
            suff = ' GB'
        elif mem >= 2**40:
            div = float(2**40)
            suff = ' TB'
        else:
            div = 1.0
            suff = ' BYTES'

        print_str = 'MEMORY USAGE: {:{w}.{p}f}'.format(process.memory_info().rss / div, w=5, p=2) + suff

        Opt.cprint(print_str)

    @staticmethod
    def time_now():
        """
        Prints current time
        :return: print to console/output
        """
        Opt.cprint('CURRENT TIME: ' + str(datetime.datetime.now()))

    @staticmethod
    def cprint(text,
               newline='\n'):
        """
        Print to stdout and flush
        :param text: Text to print to stdout
        :param newline: Newline character, default '\n' but can be changed
                        to '' to concatenate consequetive outputs
        """
        sys.stdout.write(str(text) + newline)
        sys.stdout.flush()

    @staticmethod
    def __copy__(obj):
        """
        Make a copy of an object using python's copy module
        """
        return copy.deepcopy(obj)

    @staticmethod
    def temp_name():
        """
        Generate a temperory name based on current machine time with a .tmp extension
        """
        return 'T' + datetime.datetime.now().isoformat().replace(':', '')\
            .replace('.', '').replace('-', '').replace('T', '') + '.tmp'


class FTPHandler(Handler):
    """Class to handle remote IO for ftp connection"""
    def __init__(self,
                 filename=None,
                 basename=None,
                 dirname=None,
                 ftpserv=None,
                 ftpuser=None,
                 ftppath=None,
                 ftppasswd=None,
                 ftpfilepath=None):

        super(FTPHandler, self).__init__(filename,
                                         basename,
                                         dirname)
        self.ftpserv = ftpserv
        self.ftppath = ftppath
        self.ftpuser = ftpuser
        self.ftppasswd = ftppasswd
        self.ftpfilepath = ftpfilepath
        self.conn = None

    def __repr__(self):
        if self.ftpserv is not None:
            return '<Handler for remote ftp {}>'.format(self.ftpserv)
        else:
            return '<Handler for remote ftp ____>'

    def connect(self):
        """Handle for ftp connections"""
        # define ftp
        ftp = ftplib.FTP(self.ftpserv)

        # login
        if self.ftpuser is not None:
            if self.ftppasswd is not None:
                ftp.login(user=self.ftpuser, passwd=self.ftppasswd)
            else:
                ftp.login(user=self.ftpuser)
        else:
            ftp.login()

        # get ftp connection object
        self.conn = ftp

    def disconnect(self):
        """close connection"""
        self.conn.close()

    def getfiles(self):
        """Copy all files from FTP that are in a list"""

        # connection
        ftp_conn = self.conn

        # get file(s) and write to disk
        if isinstance(self.ftpfilepath, list):
            filenamelist = [self.dirname + self.sep + Handler(ftpfile).basename
                            for ftpfile in self.ftpfilepath]

            for i in range(0, len(filenamelist)):
                self.filename = filenamelist[i]
                with open(self.filename, 'wb') as f:
                    try:
                        if ftp_conn.retrbinary("RETR {}".format(self.ftpfilepath[i]), f.write):
                            print('Copying file {} to {}'.format(self.basename,
                                                                 self.dirname))

                    except Exception as err:
                        Opt.cprint('File {} not found or already written\n Error: {}'.format(self.basename,
                                                                                             err))
        else:
            self.filename = self.dirname + Handler().sep + Handler(self.ftpfilepath).basename
            with open(self.filename, 'wb') as f:
                try:
                    if ftp_conn.retrbinary("RETR {}".format(self.ftpfilepath), f.write):
                        Opt.cprint('Copying file {} to {}'.format(Handler(self.ftpfilepath).basename,
                                                             self.dirname))
                except Exception:
                    Opt.cprint('File {} not found or already written'.format(self.basename))


class Logger(object):
    """
    Class for logging output.
    Can be instantialized to log outputs to a file or stream
    """

    def __init__(self,
                 name='default',
                 filename=None,
                 level='info',
                 stream=False):
        """
        Instantialize logger class object
        """

        self.logger = logging.getLogger(name)

        if level.lower() == 'debug':
            self.logger.setLevel(logging.DEBUG)
        elif level.lower() == 'info':
            self.logger.setLevel(logging.INFO)
        elif level.lower() == 'warn' or level.lower() == 'warning':
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.ERROR)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

        # file handler
        if filename is not None:
            file_handler = logging.FileHandler(filename,
                                               mode='w')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # stream handler
        if stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def __repr__(self):
        return '<Logger class object>'

    def lprint(self,
               message_str=None,
               level='info'):
        """
        Method to print to logger
        """

        if level.lower() == 'info':
            self.logger.info(message_str)
        elif level.lower() == 'warn' or level.lower() == 'warning':
            self.logger.warning(message_str)
        elif level.lower() == 'error':
            self.logger.error(message_str)
        else:
            self.logger.debug(message_str)

    def close(self):
        """
        Close logger
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)


class Timer:
    """
    Decorator class to compute time a function takes to execute

    example usage:

    @Timer.timing(True)
    def my_function(a, b, c):
        return a + b + c

    >Time it took to run my_function: 0.44 seconds

    """
    def __init__(self,
                 func):
        """
        Constructor
        :param func: Input function
        """
        self.func = func

    def __repr__(self):
        """
        Object representation
        :return: String
        """
        return '<Timer class for {}>'.format(self.func.__name__)

    def __call__(self, *args, **kwargs):
        """
        Function to compute timing for input function
        :return: Function and prints time taken
        """

        t1 = time.time()
        val = self.func(*args, **kwargs)
        t2 = time.time()

        # time to run
        t = self.display_time(t2 - t1)

        sys.stdout.write("Time it took to run {}: {}\n".format(self.func.__name__, t))
        return val

    @staticmethod
    def display_time(seconds,
                     precision=3):
        """
        method to display time in human readable format
        :param seconds: Number of seconds
        :param precision: Decimal precision
        :return: String
        """

        # define denominations
        intervals = [('weeks', 604800),
                     ('days', 86400),
                     ('hours', 3600),
                     ('minutes', 60),
                     ('seconds', 1)]

        # initialize list
        result = list()

        # coerce to float
        dtype = type(seconds).__name__
        if dtype != 'int' or dtype != 'long' or dtype != 'float':
            try:
                seconds = float(seconds)
            except (TypeError, ValueError, NameError):
                print("Type not coercible to Float")

        # break denominations
        for name, count in intervals:
            if name != 'seconds':
                value = seconds // count
                if value:
                    seconds -= value * count
                    if value == 1:
                        name = name.rstrip('s')
                    value = str(int(value))
                    result.append("{v} {n}".format(v=value,
                                                   n=name))
            else:
                value = "{:.{p}f}".format(seconds,
                                          p=precision)
                result.append("{v} {n}".format(v=value,
                                               n=name))

        # join output
        return ' '.join(result)

    @staticmethod
    def timing(func):
        """
        Function to compute timing for input function
        :return: Function and prints time taken
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper for func
            :param args: Arguments
            :param kwargs: Key word arguments
            :return: Function func return
            """

            t1 = time.time()
            val = func(*args, **kwargs)
            t2 = time.time()

            # time to run
            t = Timer.display_time(t2 - t1)

            sys.stdout.write("Time it took to run {}: {}\n".format(func.__name__, t))
            return val

        return wrapper

