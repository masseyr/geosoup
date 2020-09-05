import numpy as np
from geosoup.common import Handler, Opt, Sublist
from geosoup.exceptions import ImageProcessingError, ObjectNotFound
import warnings
import random
import time
import json
from osgeo import gdal, gdal_array, ogr, osr, gdalconst
np.set_printoptions(suppress=True)

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


__all__ = ['Raster',
           'MultiRaster',
           'GDAL_FIELD_DEF',
           'GDAL_FIELD_DEF_INV']


GDAL_FIELD_DEF = {
    'byte': gdal.GDT_Byte,
    'int': gdal.GDT_Int16,
    'long': gdal.GDT_Int32,
    'float': gdal.GDT_Float32,
    'double': gdal.GDT_Float64,
    'uint': gdal.GDT_UInt16,
    'ulong': gdal.GDT_UInt32,
}

GDAL_FIELD_DEF_INV = dict(list((v, k) for k, v in GDAL_FIELD_DEF.items()))

OGR_FIELD_DEF = {
    'int': ogr.OFTInteger,
    'long': ogr.OFTInteger,
    'float': ogr.OFTReal,
    'double': ogr.OFTReal,
    'str': ogr.OFTString,
    'bool': ogr.OFTInteger,
    'nonetype': ogr.OFSTNone,
    'none': ogr.OFSTNone
}

OGR_FIELD_DEF_INV = dict(list((v, k) for k, v in OGR_FIELD_DEF.items()))

OGR_TYPE_DEF = {
            'point': 1,
            'line': 2,
            'linestring': 2,
            'polygon': 3,
            'multipoint': 4,
            'multilinestring': 5,
            'multipolygon': 6,
            'geometry': 0,
            'no geometry': 100
}


class Raster(object):
    """
    Class to read and write rasters from/to files and numpy arrays
    """

    def __init__(self,
                 name,
                 array=None,
                 bnames=None,
                 metadict=None,
                 dtype=None,
                 shape=None,
                 transform=None,
                 crs_string=None,
                 datasource=None):

        self.array = array
        self.array_offsets = None  # (px, py, xoff, yoff)
        self.bnames = bnames
        self.datasource = datasource
        self.shape = shape
        self.transform = transform
        self.crs_string = crs_string
        self.name = name

        if type(dtype) == str:
            try:
                self.dtype = GDAL_FIELD_DEF[dtype]
            except Exception in (TypeError, ValueError, KeyError):
                try:
                    self.dtype = int(dtype)
                except Exception in (TypeError, ValueError, KeyError):
                    self.dtype = GDAL_FIELD_DEF['float']
        elif type(dtype) in (int, float):
            self.dtype = int(dtype)
        else:
            self.dtype = gdal.GDT_Float32

        self.metadict = metadict
        self.nodatavalue = None
        self.tile_grid = list()
        self.ntiles = None
        self.bounds = None
        self.init = False
        self.stats = dict()

    def __repr__(self):

        if self.shape is not None:
            return "<raster {ras} of size {bands}x{rows}x{cols} ".format(ras=Handler(self.name).basename,
                                                                         bands=self.shape[0],
                                                                         rows=self.shape[1],
                                                                         cols=self.shape[2]) + \
                "datatype {dt} 'no-data' value {nd}>".format(dt=str(self.dtype),
                                                             nd=str(self.nodatavalue))
        else:
            return "<raster with path {ras}>".format(ras=self.name,)

    def initialize(self,
                   get_array=False,
                   band_order=None,
                   finite_only=True,
                   nan_replacement=0.0,
                   verbose=False):

        """
        Initialize a raster object from a file
        :param get_array: flag to include raster as 3 dimensional array (bool)
        :param band_order: band location list of integers (starting at 0; ignored if get_array is False)
        :param finite_only: flag to remove non-finite values from array (ignored if get_array is False)
        :param nan_replacement: replacement for all non-finite replacements
        :param verbose: If some steps should be displayed
        (ignored if finite_only, get_array is false)
        :return None
        """
        self.init = True

        raster_name = self.name

        if Handler(raster_name).file_exists() or 'vsimem' in self.name:
            fileptr = gdal.Open(raster_name)  # open file
            self.datasource = fileptr
            self.metadict = Raster.get_raster_metadict(file_name=raster_name)

        elif self.datasource is not None:
            fileptr = self.datasource
            self.metadict = Raster.get_raster_metadict(file_ptr=fileptr)

        else:
            raise ImageProcessingError('No datasource found')

        if band_order is None:
            band_order = list(range(fileptr.RasterCount))
        elif max(band_order) >= fileptr.RasterCount:
            raise ImageProcessingError('Band indices must be smaller than number of available bands')

        self.shape = [fileptr.RasterCount if not get_array else len(band_order),
                      fileptr.RasterYSize,
                      fileptr.RasterXSize]

        if self.dtype is not None:
            self.dtype = fileptr.GetRasterBand(1).DataType

        self.nodatavalue = fileptr.GetRasterBand(1).GetNoDataValue()

        # if get_array flag is true
        if get_array:
            if verbose:
                Opt.cprint('Reading bands: {}'.format(" ".join([str(b) for b in band_order])))

            nbands = len(band_order)

            # initialize array
            if self.array_offsets is None:
                array3d = np.zeros((nbands,
                                    self.shape[1],
                                    self.shape[2]),
                                   gdal_array.GDALTypeCodeToNumericTypeCode(self.dtype))
            else:
                array3d = np.zeros((nbands,
                                    self.array_offsets[3],
                                    self.array_offsets[2]),
                                   gdal_array.GDALTypeCodeToNumericTypeCode(self.dtype))

            # get band names
            names = list()

            # read array and store the band values and name in array
            for i, b in enumerate(band_order):
                bandname = fileptr.GetRasterBand(b + 1).GetDescription()

                if verbose:
                    Opt.cprint('Reading band {}'.format(bandname))
                names.append(bandname)

                if self.array_offsets is None:
                    array3d[i, :, :] = fileptr.GetRasterBand(b + 1).ReadAsArray()
                else:
                    array3d[i, :, :] = fileptr.GetRasterBand(b + 1).ReadAsArray(*self.array_offsets)

            # if flag for finite values is present
            if finite_only:
                if np.isnan(array3d).any() or np.isinf(array3d).any():
                    array3d[np.isnan(array3d)] = nan_replacement
                    array3d[np.isinf(array3d)] = nan_replacement
                    if verbose:
                        Opt.cprint("Non-finite values replaced with " + str(nan_replacement))
                else:
                    if verbose:
                        Opt.cprint("Non-finite values absent in file")

            # assign to empty class object
            self.array = array3d
            self.bnames = names

            self.transform = fileptr.GetGeoTransform()
            self.crs_string = fileptr.GetProjection()

        else:
            self.bnames = list(fileptr.GetRasterBand(i + 1).GetDescription()
                               for i in range(self.shape[0]))
            self.transform = fileptr.GetGeoTransform()
            self.crs_string = fileptr.GetProjection()

        self.bounds = self.get_bounds()

    def write_to_file(self,
                      outfile=None,
                      driver='GTiff',
                      add_overview=False,
                      resampling='nearest',
                      overviews=None,
                      verbose=False,
                      **kwargs):
        """
        Write raster to file, given all the properties
        :param self: Raster object
        :param driver: raster driver
        :param outfile: Name of output file
        :param add_overview: If an external overview should be added to the file (useful for display)
        :param resampling: resampling type for overview (nearest, cubic, average, mode, etc.)
        :param overviews: list of overviews to compute( default: [2, 4, 8, 16, 32, 64, 128, 256])
        :param verbose: If the steps should be displayed
        :param kwargs: keyword arguments for creation options
                       example:
                       geotiff creation options can be used as follows:
                         compress='lzw',
                         bigtiff='yes
        """
        creation_options = []
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                creation_options.append('{}={}'.format(key.upper(),
                                                       value.upper()))
        if outfile is None:

            if driver == 'MEM':
                outfile = 'tmp'
            else:
                outfile = self.name
                outfile = Handler(filename=outfile).file_remove_check()

        if verbose:
            Opt.cprint('\nWriting {}\n'.format(outfile))

        gtiffdriver = gdal.GetDriverByName(driver)
        fileptr = gtiffdriver.Create(outfile, self.shape[2], self.shape[1],
                                     self.shape[0], self.dtype, creation_options)
        nbands = self.shape[0]
        fileptr.SetGeoTransform(self.transform)
        fileptr.SetProjection(self.crs_string)

        if len(self.bnames) > 0:
            for i, bname in enumerate(self.bnames):
                if len(bname) == 0:
                    self.bnames[i] = 'band_{}'.format(str(i + 1))
        else:
            for i in range(self.shape[0]):
                self.bnames[i] = 'band_{}'.format(str(i + 1))

        if self.array is None:
            self.read_array()

        for i in range(0, nbands):
            fileptr.GetRasterBand(i + 1).WriteArray(self.array[i, :, :], 0, 0)
            fileptr.GetRasterBand(i + 1).SetDescription(self.bnames[i])

            if self.nodatavalue is not None:
                fileptr.GetRasterBand(i + 1).SetNoDataValue(self.nodatavalue)
            if verbose:
                Opt.cprint('Writing band: ' + self.bnames[i])

        if driver == 'MEM':
            if verbose:
                Opt.cprint('File written to memory!')
            return fileptr

        else:
            fileptr.FlushCache()
            fileptr = None

            if verbose:
                Opt.cprint('File written to disk!')

            if add_overview:
                if verbose:
                    Opt.cprint('\nWriting overview')

                out_ras = Raster(outfile)
                out_ras.add_overviews(resampling,
                                      overviews,
                                      **kwargs)
                out_ras = None

                if verbose:
                    Opt.cprint('Overview written to disk!')

    def add_overviews(self,
                      resampling='nearest',
                      overviews=None,
                      **kwargs):
        """
        Method to create raster overviews
        :param resampling:
        :param overviews:
        :param kwargs:
        :return:
        """

        fileptr = gdal.Open(self.name, 0)

        if overviews is None:
            overviews = [2, 4, 8, 16, 32, 64, 128, 256]

        if type(overviews) not in (list, tuple):
            if type(overviews) in (str, float):
                try:
                    overviews = [int(overviews)]
                except Exception as e:
                    Opt.cprint(e.args[0])
            elif type(overviews) == int:
                overviews = [overviews]
            else:
                raise ValueError('Unsupported data type for overviews list')
        else:
            if any(list(type(elem) != int for elem in overviews)):
                overviews_ = list()
                for elem in overviews:
                    try:
                        overviews_.append(int(elem))
                    except Exception as e:
                        Opt.cprint('Conversion error: {} -for- {}'.format(e.args[1], elem))

                overviews = overviews_

        for k, v in kwargs.items():
            gdal.SetConfigOption('{}_OVERVIEW'.format(k.upper()), v.upper())

        fileptr.BuildOverviews(resampling.upper(), overviews)
        fileptr = None

    def read_array(self,
                   offsets=None,
                   band_order=None,
                   n_tries=5,
                   nodatavalue=-9999):
        """
        Method to read raster array with offsets and a specific band order
        :param offsets: tuple or list - (xoffset, yoffset, xcount, ycount)
        :param band_order: order of bands to read
        :param n_tries: number of tries after read error
        :param nodatavalue: No data value
        """

        if not self.init:
            self.initialize()

        fileptr = self.datasource

        nbands, nrows, ncols = self.shape

        if offsets is None:
            self.array_offsets = (0, 0, ncols, nrows)
        else:
            self.array_offsets = offsets

        array3d = np.zeros((nbands,
                            self.array_offsets[3],
                            self.array_offsets[2]),
                           gdal_array.GDALTypeCodeToNumericTypeCode(fileptr.GetRasterBand(1).DataType))

        # read array and store the band values and name in array
        if band_order is not None:
            for b in band_order:
                self.bnames.append(self.datasource.GetRasterBand(b + 1).GetDescription())
        else:
            band_order = list(range(nbands))

        # read array and store the band values and name in array
        for i, b in enumerate(band_order):
            read_success = True
            for i_try in range(n_tries):
                try:
                    if self.array_offsets is None:
                        array3d[i, :, :] = fileptr.GetRasterBand(b + 1).ReadAsArray()
                    else:
                        array3d[i, :, :] = fileptr.GetRasterBand(b + 1).ReadAsArray(*self.array_offsets,
                                                                                    resample_alg=
                                                                                    gdalconst.GRA_NearestNeighbour)
                    break
                except RuntimeError as e:
                    read_success = False
                    Opt.cprint(e)
                    Opt.cprint('Retrying failed read... attempt {}'.format(str(i_try + 1)))
                    time.sleep(random.random() + 1.0)

            if not read_success:
                Opt.cprint('Retrying failed with {} attempts...using no-data value'.format(str(n_tries)))

                if self.nodatavalue is not None:
                    array3d[i, :, :] = self.nodatavalue
                else:
                    array3d[i, :, :] = nodatavalue

        if (self.shape[0] == 1) and (len(array3d.shape) > 2):
            self.array = array3d.reshape([self.array_offsets[3],
                                          self.array_offsets[2]])
        else:
            self.array = array3d

    def nodata_mask(self,
                    use_band=0,
                    outfile=None,
                    in_memory=False,
                    add_overviews=False,
                    out_format='GTiff',
                    data_type=gdal.GDT_Byte):
        """
        Method to write a mask for the input raster using nodata value as 0 and valid pixels as 1
        :param use_band: Band index to use for making the mask (Default: 0)
        :param outfile: Output file path
        :param in_memory: If the raster should be stored in memory (Default: False)
        :param add_overviews: If overviews should be added,
                              ignored if in_memory is True (Default: False)
        :param out_format: output file format,
                           ignored if in_memory is True (Default: 'GTiff')
        :param data_type: Output data type (default: Byte)
        """

        self.initialize(get_array=True,
                        band_order=[use_band])

        if not in_memory:
            if outfile is None:
                outfile = Handler(self.name).add_to_filename('_mask')

        temp_arr = self.array[0, :, :]
        temp_arr[np.where(temp_arr != self.nodatavalue)] = 1
        temp_arr[np.where(temp_arr == self.nodatavalue)] = 0

        temp_ras = Raster('temp_ras')
        temp_ras.nodatavalue = self.nodatavalue
        temp_ras.array = temp_arr
        temp_ras.shape = [1, self.shape[1], self.shape[2]]
        temp_ras.transform = self.transform
        temp_ras.crs_string = self.crs_string
        temp_ras.dtype = data_type
        temp_ras.bnames = ['mask']

        if in_memory:
            temp_ras.datasource = temp_ras.write_to_file(driver='MEM')
            return temp_ras

        else:
            temp_ras.write_to_file(outfile=outfile,
                                   driver=out_format,
                                   add_overviews=add_overviews)

    def set_nodataval(self,
                      in_nodataval=255,
                      out_nodataval=0,
                      outfile=None,
                      in_array=True,
                      **kwargs):
        """
        replace no data value in raster and write to tiff file
        :param in_nodataval: no data value in input raster
        :param out_nodataval: no data value in output raster
        :param in_array: if the no data value should be changed in raster array
        :param outfile: output file name
        """
        if in_array:
            if not self.init:
                self.initialize(get_array=True,
                                **kwargs)
            self.array[np.where(self.array == in_nodataval)] = out_nodataval

        self.nodatavalue = out_nodataval

        if outfile is not None:
            self.write_to_file(outfile)

    @property
    def chk_for_empty_tiles(self):
        """
        check the tile for empty bands, return true if one exists
        :return: bool
        """
        if Handler(self.name).file_exists():
            fileptr = gdal.Open(self.name)

            filearr = fileptr.ReadAsArray()
            nb, ns, nl = filearr.shape

            truth_about_empty_bands = [np.isfinite(filearr[i, :, :]).any() for i in range(0, nb)]

            fileptr = None

            return any([not x for x in truth_about_empty_bands])
        else:
            raise ObjectNotFound("File does not exist.")

    @staticmethod
    def get_raster_metadict(file_name=None,
                            file_ptr=None):
        """
        Function to get all the spatial metadata associated with a geotiff raster
        :param file_name: Name of the raster file (includes full path)
        :param file_ptr: Gdal file pointer
        :return: Dictionary of raster metadata
        """
        if file_name is not None:
            if Handler(file_name).file_exists():
                # open raster
                img_pointer = gdal.Open(file_name)
            else:
                raise ObjectNotFound("File does not exist.")

        elif file_ptr is not None:
            img_pointer = file_ptr

        else:
            raise ObjectNotFound("File or pointer not found")

        # get tiepoint, pixel size, pixel rotation
        geometadata = img_pointer.GetGeoTransform()

        # make dictionary of all the metadata
        meta_dict = {'ulx': geometadata[0],
                     'uly': geometadata[3],
                     'xpixel': abs(geometadata[1]),
                     'ypixel': abs(geometadata[5]),
                     'rotationx': geometadata[2],
                     'rotationy': geometadata[4],
                     'datatype': img_pointer.GetRasterBand(1).DataType,
                     'columns': img_pointer.RasterXSize,  # columns from raster pointer
                     'rows': img_pointer.RasterYSize,  # rows from raster pointer
                     'bands': img_pointer.RasterCount,  # bands from raster pointer
                     'projection': img_pointer.GetProjection(),  # projection information from pointer
                     'name': Handler(file_name).basename}  # file basename

        # remove pointer
        img_pointer = None

        return meta_dict

    def get_stats(self,
                  verbose=False,
                  approx=False):

        """
        Method to compute statistics of the raster object, and store as raster property
        :param verbose: If the statistics should be printed to console
        :param approx: If approx statistics should be calculated instead to gain speed
        :return: None
        """

        for ib in range(self.shape[0]):
            band = self.datasource.GetRasterBand(ib+1)
            band.ComputeStatistics(approx)
            band_stats = dict(zip(['min', 'max', 'mean', 'stddev'], band.GetStatistics(int(approx), 0)))

            if verbose:
                Opt.cprint('Band {} : {}'.format(self.bnames[ib],
                                                 str(band_stats)))

            self.stats[self.bnames[ib]] = band_stats

    def rescale(self,
                out_type=None,
                rescale=False,
                scale_vals=None,
                scale_approx=False):

        """
        Method to rescale raster or to change the raster data type
        :param out_type: Out data type.
                         Valid options: byte, int, long, float, double
                                        int8, int16, int32, int64,
                                        float32, float64,
                                        uint8, uint16, uint32
        :param rescale: If the image should be rescaled
        :param scale_vals: Tuple of min and max values for rescaling
        :param scale_approx: If approx stats should be calculated
        :return: None
        """

        if out_type is None:
            out_type = gdal_array.GDALTypeCodeToNumericTypeCode(self.dtype)
        else:
            common_types = {'byte': 'uint8',
                            'int': 'int16',
                            'float': 'float32',
                            'double': 'float64',
                            'long': 'int32'}

            if out_type in common_types:
                out_type = common_types[out_type]

        if gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype(out_type)) != self.dtype:

            if rescale:
                if scale_vals is not None:

                    if self.stats is None:
                        self.get_stats(approx=scale_approx)
                    elif len(self.stats) < self.shape[0]:
                        self.get_stats(approx=scale_approx)

                    out_arr = np.zeros(self.shape, dtype=out_type)

                    for band_indx in range(self.shape[0]):
                        min_val, max_val = self.stats[band_indx]['min'], self.stats[band_indx]['max']

                        mdiff = float(scale_vals[1] - scale_vals[0]) / float(max_val - min_val)

                        out_arr[band_indx, :, :] = ((self.array[band_indx, :, :].astype(np.float64) - min_val) * mdiff +
                                                    scale_vals[0]).astype(out_type)

                    self.array = out_arr
                else:
                    self.array = self.array.astype(out_type)
            else:
                self.array = self.array.astype(out_type)

            self.dtype = gdal_array.NumericTypeCodeToGDALTypeCode(self.array.dtype)

            if self.nodatavalue is not None:
                self.nodatavalue = np.array(self.nodatavalue).astype(out_type).item()

    def make_polygon_geojson_feature(self):
        """
        Make a feature geojson for the raster using its metaDict data
        """

        meta_dict = self.metadict

        if meta_dict is not None:
            return {"type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                             [meta_dict['ulx'], meta_dict['uly']],
                             [meta_dict['ulx'], meta_dict['uly'] - (meta_dict['ypixel'] * (meta_dict['rows'] + 1))],
                             [meta_dict['ulx'] + (meta_dict['xpixel'] * (meta_dict['columns'] + 1)),
                              meta_dict['uly'] - (meta_dict['ypixel'] * (meta_dict['rows'] + 1))],
                             [meta_dict['ulx'] + (meta_dict['xpixel'] * (meta_dict['columns'] + 1)), meta_dict['uly']],
                             [meta_dict['ulx'], meta_dict['uly']]
                             ]]
                        },
                    "properties": {
                        "name": meta_dict['name'].split('.')[0]
                        },
                    }
        else:
            raise AttributeError("Metadata dictionary does not exist.")

    @staticmethod
    def get_coords(xy_list,
                   pixel_size,
                   tie_point,
                   pixel_center=True):
        """
        Method to convert pixel locations to image coords
        :param xy_list: List of tuples [(x1,y1), (x2,y2)....]
        :param pixel_size: tuple of x and y pixel size. The signs of the pixel sizes (+/-) are as in GeoTransform
        :param tie_point: tuple of x an y coordinates of tie point for the xy list
        :param pixel_center: If the center of the pixels should be returned instead of the top corners (default: True)
        :return: List of coordinates in tie point coordinate system
        """

        if type(xy_list) != list:
            xy_list = [xy_list]

        add_const = (float(pixel_size[0])/2.0) * pixel_center, \
                    (float(pixel_size[1])/2.0) * pixel_center

        return list((float(xy[0]) * float(pixel_size[0]) + tie_point[0] + add_const[0],
                     float(xy[1]) * float(pixel_size[1]) + tie_point[1] + add_const[1])
                    for xy in xy_list)

    @staticmethod
    def get_locations(coords_list,
                      pixel_size,
                      tie_point):
        """
        Method to convert global coordinates to image pixel locations
        :param coords_list: Lit of coordinates in image CRS [(x1,y1), (x2,y2)....]
        :param pixel_size: Pixel size
        :param tie_point: Tie point of the raster or tile
        :return: list of pixel locations
        """
        if type(coords_list) != list:
            coords_list = [coords_list]

        return list(((coord[0] - tie_point[0]) // pixel_size[0],
                     (coord[1] - tie_point[1]) // pixel_size[1])
                    if coord is not None else [None, None]
                    for coord in coords_list)

    def get_bounds(self,
                   xy_coordinates=True,
                   use_nodatavalue=False,
                   return_datasource=False,
                   return_memory_dataset=False,
                   with_buffer=0.0):
        """
        Method to return a list of raster coordinates
        :param xy_coordinates: return a list of xy coordinates if true,
                              else return [xmin, xmax, ymin, ymax]
                              Ignored if use_nodatavalue is True
        :param use_nodatavalue: If the nodatavalue should be used to vectorize areas with data
                                This will return a wkt geometry
        :param return_datasource: if bounds data sorce object should be returned
        :param return_memory_dataset: If a dataset in memory should be returned (/vsimem object)
        :param with_buffer: the geometry is buffered by specified pixels if > 0
        :return:
              if use_nodatavalue flag is False and xy_coordinates is False: list of numbers
              if use_nodatavalue is False and xy_coordinates is True: list of lists

              if use_nodatavalue is True: wkt_geometry
              if return_memory_dataset is selected: returns /vsimem/{basename}_bounds.shp
        """
        if not self.init:
            self.initialize()
        tie_pt = [self.transform[0], self.transform[3]]

        if not use_nodatavalue:
            if xy_coordinates:
                return [tie_pt,
                        [tie_pt[0] + self.metadict['xpixel'] * self.shape[2], tie_pt[1]],
                        [tie_pt[0] + self.metadict['xpixel'] * self.shape[2],
                         tie_pt[1] - self.metadict['ypixel'] * self.shape[1]],
                        [tie_pt[0], tie_pt[1] - self.metadict['ypixel'] * self.shape[1]],
                        tie_pt]
            else:
                return [tie_pt[0], tie_pt[0] + self.metadict['xpixel'] * self.shape[2],
                        tie_pt[1] - self.metadict['ypixel'] * self.shape[1], tie_pt[1]]
        else:

            mask_ras = self.nodata_mask(in_memory=True)
            basename = Handler(self.name).basename.split('.')[0]
            out_name = None

            if return_memory_dataset:
                out_name = '/vsimem/{}_bounds.shp'.format(basename)
                out_driver = ogr.GetDriverByName('ESRI Shapefile')
                out_datasource = out_driver.CreateDataSource('/vsimem/{}_bounds.shp'.format(basename))

            else:
                out_driver = ogr.GetDriverByName('Memory')
                out_datasource = out_driver.CreateDataSource('mem_source')

            out_spref = osr.SpatialReference()
            out_spref.ImportFromWkt(self.crs_string)

            out_layer = out_datasource.CreateLayer('{}_bounds'.format(basename),
                                                   srs=out_spref,
                                                   geom_type=OGR_TYPE_DEF['multipolygon'])

            out_field = ogr.FieldDefn('tmp_field', OGR_FIELD_DEF['int'])
            out_layer.CreateField(out_field)

            gdal.Polygonize(mask_ras.datasource.GetRasterBand(1),
                            None,
                            out_layer,
                            0)

            out_datasource.FlushCache()

            geom_wkt = None
            del_feat_id = []
            nfeat = out_layer.GetFeatureCount()

            for feat_indx in range(nfeat):
                feat = out_layer.GetFeature(feat_indx)
                items = feat.items()
                if items['tmp_field'] != 1:
                    out_layer.DeleteFeature(feat.GetFID())
                else:
                    geom = feat.GetGeometryRef()
                    geom.CloseRings()
                    if with_buffer > 0:
                        geom = geom.Buffer(self.transform[0]*float(with_buffer))

                    geom_wkt = geom.ExportToWkt()

            out_datasource.ExecuteSQL('REPACK ' + out_layer.GetName())
            out_datasource.ExecuteSQL('RECOMPUTE EXTENT ON ' + out_layer.GetName())

            if return_datasource:
                return out_datasource

            elif return_memory_dataset:
                out_datasource = None
                return out_name

            else:

                mask_ras = None
                out_datasource = None
                return geom_wkt

    def centroid(self,
                 use_nodatavalue=False):
        """
        Method to return centroid of the Raster object based on its boundary
        :param use_nodatavalue: Boolean flag to calculate the centroid
                                based on valid pixel boundary
        """
        bounds = self.get_bounds(use_nodatavalue=use_nodatavalue,
                                 xy_coordinates=False)

        if use_nodatavalue:
            bounds_geom = ogr.CreateGeometryFromWkt(bounds)
            centroid_geom = bounds_geom.Centroid()
            return centroid_geom.GetPoint()[0:2]

        else:
            bounds = self.get_bounds()
            return float(bounds[0] + bounds[1])/2.0, float(bounds[2] + bounds[3])/2.0

    def get_pixel_bounds(self,
                         bound_coords=None,
                         coords_type='pixel'):
        """
        Method to return image bounds in the format xmin, xmax, ymin, ymax
        :param bound_coords: (xmin, xmax, ymin, ymax)
        :param coords_type: type of coordinates specified in bound_coords: 'pixel' for pixel coordinates,
                                                                           'crs' for image reference system coordinates
        :return: tuple: (xmin, xmax, ymin, ymax) in pixel coordinates
        """
        if not self.init:
            self.initialize()

        if bound_coords is not None:
            if coords_type == 'pixel':
                xmin, xmax, ymin, ymax = bound_coords
            elif coords_type == 'crs':
                _xmin, _xmax, _ymin, _ymax = bound_coords
                coords_list = [(_xmin, _ymax), (_xmax, _ymax), (_xmax, _ymin), (_xmin, _ymin)]
                coords_locations = np.array(self.get_locations(coords_list,
                                                               (self.transform[1], self.transform[5]),
                                                               (self.transform[0], self.transform[3])))
                xmin, xmax, ymin, ymax = \
                    int(coords_locations[:, 0].min()), \
                    int(coords_locations[:, 0].max()), \
                    int(coords_locations[:, 1].min()), \
                    int(coords_locations[:, 1].max())
            else:
                raise ImageProcessingError("Unknown coordinate types")

            if xmin < 0:
                xmin = 0
            if xmax > self.shape[2]:
                xmax = self.shape[2]
            if ymin < 0:
                ymin = 0
            if ymax > self.shape[1]:
                ymax = self.shape[1]

            if xmin >= xmax:
                raise ImageProcessingError("Image x-size should be greater than 0")
            if ymin >= ymax:
                raise ImageProcessingError("Image y-size should be greater than 0")
        else:
            xmin, xmax, ymin, ymax = 0, self.shape[2], 0, self.shape[1]

        return xmin, xmax, ymin, ymax

    def make_tile_grid(self,
                       tile_xsize=1024,
                       tile_ysize=1024,
                       bound_coords=None,
                       coords_type='pixel',
                       tile_buffer=None):
        """
        Returns the coordinates of the blocks to be extracted
        :param tile_xsize: Number of columns in the tile block
        :param tile_ysize: Number of rows in the tile block
        :param bound_coords: (xmin, xmax, ymin, ymax)
        :param coords_type: type of coordinates specified in bound_coords: 'pixel' for pixel coordinates,
                                                                           'crs' for image reference system coordinates
        :param tile_buffer: Buffer outside the tile boundary in image projection units
        :return: list of lists
        """
        if not self.init:
            self.initialize()

        self.tile_grid = []

        # convert to the number of pixels in the buffer region
        if tile_buffer is not None:
            buf_size_x = np.ceil(float(tile_buffer)/abs(float(self.transform[1])))
            buf_size_y = np.ceil(float(tile_buffer)/abs(float(self.transform[5])))
        else:
            buf_size_x, buf_size_y = None, None

        xmin, xmax, ymin, ymax = self.get_pixel_bounds(bound_coords,
                                                       coords_type)

        for y in range(ymin, ymax, tile_ysize):

            if y + tile_ysize < ymax:
                rows = tile_ysize
            else:
                rows = ymax - y

            for x in range(xmin, xmax, tile_xsize):
                if x + tile_xsize < xmax:
                    cols = tile_xsize
                else:
                    cols = xmax - x

                tie_pt = self.get_coords([(x, y)],
                                         (self.transform[1], self.transform[5]),
                                         (self.transform[0], self.transform[3]),
                                         pixel_center=False)[0]

                bounds = [tie_pt,
                          [tie_pt[0] + self.transform[1] * cols, tie_pt[1]],
                          [tie_pt[0] + self.transform[1] * cols, tie_pt[1] + self.transform[5] * rows],
                          [tie_pt[0], tie_pt[1] + self.transform[5] * rows],
                          tie_pt]

                self.tile_grid.append({'block_coords': (x, y, cols, rows),
                                       'tie_point': tie_pt,
                                       'bound_coords': bounds,
                                       'first_pixel': (xmin, ymin),
                                       'edge_buffer': (buf_size_x, buf_size_y)})

        self.ntiles = len(self.tile_grid)

    def get_tile(self,
                 bands=None,
                 block_coords=None,
                 finite_only=True,
                 edge_buffer=None,
                 nan_replacement=None,
                 n_tries=5,
                 nodatavalue=-9999):
        """
        Method to get raster numpy array of a tile
        :param bands: bands to get in the array, index starts from one. (default: all)
        :param finite_only:  If only finite values should be returned
        :param edge_buffer: Number of extra pixels to retrieve further out from the edges (default: None)
                           Tuple (buf_size_x, buf_size_y)
        :param nan_replacement: replacement for NAN values
        :param n_tries: Number of randomized tries when reading raster is errored out
        :param nodatavalue: No data value default -9999
        :param block_coords: coordinates of tile to retrieve in image/array coords
                             format is (upperleft_x, upperleft_y, tile_cols, tile_rows)
                             upperleft_x and upperleft_y are array coordinates starting at 0,
                             cols and rows are number of pixels to retrieve for the tile along x and y respectively
        :return: numpy array
        """

        if edge_buffer is None:
            edge_buffer_x, edge_buffer_y =  0, 0
        else:
            edge_buffer_x, edge_buffer_y = edge_buffer

        if not self.init:
            self.initialize()

        if nan_replacement is None:
            if self.nodatavalue is None:
                nan_replacement = 0
            else:
                nan_replacement = self.nodatavalue

        if bands is None:
            bands = list(range(1, self.shape[0] + 1))

        if block_coords is None:
            raise ImageProcessingError("Block coords needed to retrieve tile")
        else:
            upperleft_x, upperleft_y, tile_rows, tile_cols = block_coords

        # raster shape param
        ras_rows, ras_cols = self.shape[1], self.shape[2]

        # accounting for number of pixels less than the required size (always >= 0)
        # pixel deficit on left, top, right, and bottom edges respectively
        pixel_deficit = [(edge_buffer_x - upperleft_x) if (upperleft_x < edge_buffer_x) else 0,

                         (edge_buffer_y - upperleft_y) if (upperleft_y < edge_buffer_y) else 0,

                         (ras_cols - upperleft_x - tile_cols + 1) if
                         (ras_cols - upperleft_x - tile_cols + 1) < edge_buffer_x else 0,

                         (ras_rows - upperleft_y - tile_rows + 1) if
                         (ras_rows - upperleft_y - tile_rows + 1) < edge_buffer_y else 0]

        # corners
        new_upperleft_x = (upperleft_x - edge_buffer_x) + pixel_deficit[0]
        new_upperleft_y = (upperleft_y - edge_buffer_y) + pixel_deficit[1]

        # new block coordinates
        new_block_coords = [new_upperleft_x,
                            new_upperleft_y,
                            tile_rows + (2 * edge_buffer_y - pixel_deficit[1] + pixel_deficit[3]),
                            tile_cols + (2 * edge_buffer_x - pixel_deficit[0] + pixel_deficit[2])]

        tile_arr = np.zeros((len(bands),
                             new_block_coords[3],
                             new_block_coords[2]),
                            gdal_array.GDALTypeCodeToNumericTypeCode(self.dtype))

        for jj, band in enumerate(bands):
            read_success = True
            for i_try in range(n_tries):
                try:
                    temp_band = self.datasource.GetRasterBand(band)
                    tile_arr[jj, :, :] = temp_band.ReadAsArray(*new_block_coords)
                    break
                except RuntimeError as e:
                    read_success = False
                    Opt.cprint(e)
                    Opt.cprint('Retrying failed read...attempt {}'.format(i_try + 1))
                    time.sleep(random.random() + 1.0)

            if not read_success:
                Opt.cprint('Retrying failed with {} attempts...using no-data value'.format(str(n_tries)))
                if self.nodatavalue is not None:
                    tile_arr[jj, :, :] = self.nodatavalue
                else:
                    tile_arr[jj, :, :] = nodatavalue

        if finite_only:
            if np.isnan(tile_arr).any() or np.isinf(tile_arr).any():
                tile_arr[np.isnan(tile_arr)] = nan_replacement
                tile_arr[np.isinf(tile_arr)] = nan_replacement

        return tile_arr

    def get_next_tile(self,
                      tile_xsize=1024,
                      tile_ysize=1024,
                      bands=None,
                      get_array=True,
                      finite_only=True,
                      edge_buffer=None,
                      nan_replacement=None):

        """
        Generator to extract raster tile by tile
        :param tile_xsize: Number of columns in the tile block
        :param tile_ysize: Number of rows in the tile block
        :param bands: List of bands to extract (default: None, gets all bands; Index starts at 0)
        :param get_array: If raster array should be retrieved as well
        :param finite_only: If only finite values should be returned
        :param edge_buffer: Tuple of extra pixels to retrieve further out from the edges (default: None)
                            (edge_buf_x, edge_buf_y)
        :param nan_replacement: replacement for NAN values
        :return: Yields tuple: (tiepoint xy tuple, tile numpy array(2d array if only one band, else 3d array)
        """

        if not self.init:
            self.initialize()

        if self.ntiles is None:
            self.make_tile_grid(tile_xsize,
                                tile_ysize)
        if nan_replacement is None:
            if self.nodatavalue is None:
                nan_replacement = 0
            else:
                nan_replacement = self.nodatavalue

        if bands is None:
            bands = range(1, int(self.shape[0]) + 1)
        elif type(bands) in (int, float):
            bands = [int(bands)]
        elif type(bands) in (list, tuple):
            if all(list(isinstance(elem, str) for elem in bands)):
                bands = [self.bnames.index(elem) for elem in bands]
            elif all(list(isinstance(elem, int) or isinstance(elem, float) for elem in bands)):
                bands = [int(ib + 1) for ib in bands]
        else:
            raise ValueError('Unknown/unsupported data type for "bands" keyword')

        tile_counter = 0
        while tile_counter < self.ntiles:
            if get_array:
                tile_arr = self.get_tile(bands=bands,
                                         block_coords=self.tile_grid[tile_counter]['block_coords'],
                                         finite_only=finite_only,
                                         edge_buffer=edge_buffer,
                                         nan_replacement=nan_replacement)
            else:
                tile_arr = None

            yield self.tile_grid[tile_counter]['tie_point'], tile_arr

            tile_counter += 1

    def make_tiles(self,
                   tile_size_x=1024,
                   tile_size_y=1024,
                   out_path=None,
                   edge_buffer=None,
                   **kwargs):

        """
        Make and write smaller tile tif files from the raster tif file
        :param tile_size_y: Tile size along x
        :param tile_size_x: tile size along y
        :param edge_buffer: Buffer along the edges to include while making tiles
                            Tuple of (edge_buf_x, edge_buf_y) in number of pixels
        :param out_path: Output folder for the tiles
        :returns: None
        """
        if out_path is None:
            out_path = Handler(self.name).dirname

        self.make_tile_grid(tile_size_x, tile_size_y)

        tile_counter = 0
        for tie_pt, tile_arr in self.get_next_tile(edge_buffer=edge_buffer):
            block_coords = self.tile_grid[tile_counter]['block_coords']
            out_file = out_path + Handler(Handler(self.name).basename)\
                .add_to_filename('_{}'.format('_'.join([str(elem) for elem in block_coords])))

            out_transform = [tie_pt[0],
                             self.transform[1],
                             self.transform[2],
                             tie_pt[1],
                             self.transform[4],
                             self.transform[5]]

            tile_ras = Raster(out_file,
                              array=tile_arr,
                              bnames=self.bnames,
                              dtype=self.dtype,
                              shape=tile_arr.shape,
                              transform=out_transform,
                              crs_string=self.crs_string)

            tile_ras.write_to_file(**kwargs)
            tile_ras = None
            tile_counter += 1

    @staticmethod
    def index_geom_wkt(wkt_strings,
                       geom_ids=None,
                       separate_multi_geom=True):

        """
        Method to index list of OGR geometry strings and
        split OGR multi-geometry wkt strings into single geometries

        :param wkt_strings: List of OGR geometry wkt strings
        :param geom_ids: List of geometry ids corresponding to wkt_string list
        :param separate_multi_geom: (bool) Flag to separate multigeometries into smaller geometries
        :returns: List of tuples [(geom_id: OGR geometry obj), ]
        """

        id_geom_list = []

        for wkt_string_indx, wkt_string in enumerate(wkt_strings):
            # multi geometry should be separated
            if separate_multi_geom:
                if ('MULTI' in wkt_string) or ('multi' in wkt_string):

                    # if multi geometry should be separated then add M prefix to index and
                    # add another index of the geometry after underscore
                    multi_geom = ogr.CreateGeometryFromWkt(wkt_string)

                    for multi_geom_indx in range(multi_geom.GetGeometryCount()):
                        geom_internal_id = '{}_{}'.format(str(wkt_string_indx), str(multi_geom_indx))\
                            if geom_ids is None else '{}_{}'.format(str(geom_ids[wkt_string_indx]),
                                                                    str(multi_geom_indx))
                        id_geom_list.append((geom_internal_id,
                                             multi_geom.GetGeometryRef(multi_geom_indx)))

                else:
                    # if no multi geometry in the string
                    id_geom_list.append((str(wkt_string_indx) if geom_ids is None else geom_ids[wkt_string_indx],
                                         ogr.CreateGeometryFromWkt(wkt_string)))

            else:
                # if multi geometry should not be separated
                id_geom_list.append((str(wkt_string_indx) if geom_ids is None else geom_ids[wkt_string_indx],
                                     ogr.CreateGeometryFromWkt(wkt_string)))

        return id_geom_list

    def extract_geom(self,
                     wkt_strings,
                     geom_ids=None,
                     band_order=None,
                     **kwargs):
        """
        Extract all pixels that intersect a geometry or a list of geometries in a Raster.
        The raster object should be initialized before using this method.
        Currently this method only supports single geometries per query.
        :param wkt_strings: List or Tuple of Vector geometries (e.g. point) in WKT string format
                           this geometry should be in the same CRS as the raster
                           Currently only 'Point' or 'MultiPoint' geometry is supported.
                           Accepted wkt_strings: List of POINT or MULTIPOINT wkt(s)
        :param geom_ids: List of geometry IDs
                        If for a MultiGeom only one ID is presented,
                        it will be suffixed with the order of the geometry part
        :param band_order: Order of bands to be extracted (list of band indices)  default: all bands

        :param kwargs: List of additional arguments
                        tile_size : tuple (x size, y size)
                                   Size of internal tiling
                        multi_geom_separate: bool (False) default
                                            if multi geometries should be separated or not
                        pass_pixel_coords: bool (False) default
                                           if coordinates of the pixels should be passed along the output
                        pixel_center: bool (True) default
                                     Used only if the pass_pixel_coords flag is set to true.
                                     Returns the pixel centers of each pixel if set as true
                                     else returns top left corners
                        min_pixels: Minimum number of pixels for reducer to calculate a value.
                                    If the number of pixels are less than this, then an empty list is returned
                        replace: If the pixel values in Raster.array object should be replaced with
                                 the reduced value? This will only be computed of min_pixels condition is
                                 satisfied and will change the Raster array. Use with caution.
                        reducer: Valid keywords: 'mean','median','max',
                                                 'min', 'percentile_xx' where xx is percentile from 1-99

        :return: Dictionary of dictionaries
                {internal_id: {'values': [[band1, ], ], 'coordinates': [(x1, y1), ]}, }
                if geom_id is supplied then internal_id is the supplied geom_id

        todo: Implement separation of overlapping feature extractions

        """
        if band_order is None:
            band_order = list(range(self.shape[0]))

        # define tile size
        if 'tile_size' in kwargs:
            tile_size = kwargs['tile_size']
        else:
            tile_size = (self.shape[1], self.shape[2])

        # if multi geometries should be separated or not
        if 'separate_multi_geom' in kwargs:
            separate_multi_geom = kwargs['separate_multi_geom']
        else:
            separate_multi_geom = False

        if 'pass_pixel_coords' in kwargs:
            pass_pixel_coords = kwargs['pass_pixel_coords']
        else:
            pass_pixel_coords = False

        if 'pixel_center' in kwargs:
            pixel_center = kwargs['pixel_center']
        else:
            pixel_center = True

        if 'reducer' in kwargs:
            reducer = kwargs['reducer']
        else:
            reducer = None

        if 'min_pixels' in kwargs:
            min_pixels = kwargs['min_pixels']
        else:
            min_pixels = None

        if 'replace' in kwargs:
            replace = kwargs['replace']
        else:
            replace = False

        # define band order
        if band_order is None:
            band_order = list(range(0, self.shape[0]))

        # initialize raster
        if not self.init or self.array is None:
            self.initialize()

        # make wkt strings into a list
        if type(wkt_strings) not in (list, tuple):
            wkt_strings = [wkt_strings]

        # list of geometry indices and OGR SWIG geometry objects
        # each dict entry contains   geom_id : geom
        id_geom_list = self.index_geom_wkt(wkt_strings,
                                           geom_ids=geom_ids,
                                           separate_multi_geom=separate_multi_geom)

        # make internal tiles
        self.make_tile_grid(*tile_size)

        # prepare dict struct
        out_geom_extract = dict()
        for internal_id, _ in id_geom_list:
            out_geom_extract[internal_id] = {'values': [], 'coordinates': [], 'xyz_loc': []}

        geom_id_order = list(internal_id for internal_id, _ in id_geom_list)

        # list of sample ids
        for tile in self.tile_grid:

            # create tile geometry from bounds
            tile_geom = ogr.CreateGeometryFromWkt('POLYGON(({}))'.format(', '.join(list(' '.join([str(x), str(y)])
                                                                                        for (x, y)
                                                                                        in tile['bound_coords']))))

            tile_arr = self.get_tile(block_coords=tile['block_coords'])

            # check if the geometry intersects and
            # place all same geometry types together
            geom_by_type = {}
            for samp_id, samp_geom in id_geom_list:
                if tile_geom.Intersects(samp_geom):
                    geom_type = samp_geom.GetGeometryType()
                    if geom_type not in geom_by_type:
                        geom_by_type[geom_type] = [(samp_id, samp_geom)]
                    else:
                        geom_by_type[geom_type].append((samp_id, samp_geom))

            # check is any geoms are available
            if len(geom_by_type) > 0:
                for geom_type, geom_list in geom_by_type.items():

                    # get tile shape and tie point
                    _, _, rows, cols = tile['block_coords']
                    tie_pt_x, tie_pt_y = tile['tie_point']

                    # create tile empty raster in memory
                    target_ds = gdal.GetDriverByName('MEM').Create('tmp',
                                                                   cols,
                                                                   rows,
                                                                   1,
                                                                   gdal.GDT_UInt16)

                    # set pixel size and tie point
                    target_ds.SetGeoTransform((tie_pt_x,
                                               self.transform[1],
                                               0,
                                               tie_pt_y,
                                               0,
                                               self.transform[5]))

                    # set raster projection
                    target_ds.SetProjection(self.crs_string)

                    # create vector in memory
                    burn_driver = ogr.GetDriverByName('Memory')
                    burn_datasource = burn_driver.CreateDataSource('mem_source')
                    burn_spref = osr.SpatialReference()
                    burn_spref.ImportFromWkt(self.crs_string)
                    burn_layer = burn_datasource.CreateLayer('tmp_lyr',
                                                             srs=burn_spref,
                                                             geom_type=geom_type)

                    # attributes
                    fielddefn = ogr.FieldDefn('fid', ogr.OFTInteger)
                    result = burn_layer.CreateField(fielddefn)
                    layerdef = burn_layer.GetLayerDefn()

                    geom_burn_val = 1
                    geom_dict = {}
                    for geom_id, geom in geom_list:
                        # create features in layer
                        temp_feature = ogr.Feature(layerdef)
                        temp_feature.SetGeometry(geom)
                        temp_feature.SetField('fid', geom_burn_val)
                        burn_layer.CreateFeature(temp_feature)
                        geom_dict[geom_burn_val] = geom_id
                        geom_burn_val += 1

                    gdal.RasterizeLayer(target_ds,
                                        [1],
                                        burn_layer,
                                        None,  # transformer
                                        None,  # transform
                                        [1],
                                        ['ALL_TOUCHED=TRUE',
                                         'ATTRIBUTE=FID'])

                    # read mask band as array
                    temp_band = target_ds.GetRasterBand(1)
                    mask_arr = temp_band.ReadAsArray()

                    min_burn_val = min(list(geom_dict.keys()))
                    max_burn_val = max(list(geom_dict.keys()))

                    pixel_xy_loc = np.where((mask_arr >= min_burn_val) & (mask_arr <= max_burn_val))
                    burn_vals = mask_arr[pixel_xy_loc]
                    pixel_xy_loc_list = np.array(list(zip(*pixel_xy_loc)))

                    burn_dict = {}
                    for geom_burn_val in range(min_burn_val, max_burn_val + 1):
                        burn_dict[geom_burn_val] = pixel_xy_loc_list[np.where(burn_vals == geom_burn_val)].tolist()

                    for geom_burn_val, geom_id in geom_dict.items():
                        # make list of unmasked pixels
                        if pass_pixel_coords:
                            # get coordinates
                            out_geom_extract[geom_id]['coordinates'] += self.get_coords(burn_dict[geom_burn_val],
                                                                                        (self.transform[1],
                                                                                         self.transform[5]),
                                                                                        tile['tie_point'],
                                                                                        pixel_center)

                        # get band values from tile array
                        out_geom_extract[geom_id]['values'] += list(tile_arr[band_order, x, y].tolist()
                                                                    for x, y in burn_dict[geom_burn_val])

                        out_geom_extract[geom_id]['xyz_loc'] += list([[band_index] + list(xy_loc)]
                                                                     for xy_loc in burn_dict[geom_burn_val]
                                                                     for band_index in band_order)

            if reducer is not None:
                warned = False
                for geom_id, geom_dict in out_geom_extract.items():
                    try:
                        n_pixels = len(geom_dict['values'])
                        if min_pixels is None:
                            min_pixels = n_pixels

                        if n_pixels >= min_pixels:
                            geom_dict['values'] = Sublist.reduce(geom_dict['values'],
                                                                 method=reducer,
                                                                 axis=0).tolist()

                            if replace:
                                pixel_loc = (np.array(geom_dict['xyz_loc'])[:, 0],
                                             np.array(geom_dict['xyz_loc'])[:, 1],
                                             np.array(geom_dict['xyz_loc'])[:, 2])

                                self.array[pixel_loc] = geom_dict['values']

                        else:
                            geom_dict['values'] = []

                        geom_dict['coordinates'] = Sublist.reduce(geom_dict['coordinates'],
                                                                  method=reducer,
                                                                  axis=0).tolist()
                    except Exception as e:
                        if not warned:
                            warnings.warn(e.args[0] + '\nNot reducing.'.format(reducer))
                            warned = True

                    out_geom_extract[geom_id] = geom_dict

        return list(out_geom_extract[internal_id] for internal_id in geom_id_order)

    def clip(self,
             cutline_file=None,
             cutline_blend=0,
             outfile=None,
             return_vrt=False,
             return_vrt_dict=False,
             cutline_as_mask=True,
             **creation_options):
        """
        Method to clip a raster to a given geometry/vector
        This method only supports clipping to the first layer in the datasource
        :param cutline_file: Shapefile, etc. to clip raster
        :param cutline_blend: blend distance in pixels
        :param outfile: Output filename
        :param return_vrt: If a VRT object should be returned instead of raster
        :param return_vrt_dict: if a VRT options dictionary should be returned instead
        :param cutline_as_mask: Use cutline extent for output bounds (default: true)
        :param creation_options: Other creation options to input
        :return: Raster object

        valid warp options can be found at:
        https://gdal.org/python/osgeo.gdal-module.html#WarpOptions:
        """
        cutline_ds = ogr.Open(cutline_file)
        layer_count = cutline_ds.GetLayerCount()
        cutline_layer = cutline_ds.GetLayer(0)
        cutline_layer_name = cutline_layer.GetDescription()

        if layer_count > 1:
            warnings.warn('Using top layer {} as cutline, ignoring others'.format(cutline_layer_name))

        cutline_ds = cutline_layer = None

        vrt_dict = dict()

        vrt_dict['cutlineDSName'] = cutline_file
        vrt_dict['cutlineLayer'] = cutline_layer_name
        vrt_dict['cutlineBlend'] = cutline_blend
        vrt_dict['cropToCutline'] = cutline_as_mask
        vrt_dict['copyMetadata'] = True

        creation_options_list = []
        if len(creation_options) > 0:
            for key, value in creation_options.items():
                creation_options_list.append('{}={}'.format(key.upper(),
                                                            value.upper()))
            vrt_dict['creationOptions'] = creation_options_list

        vrt_opt = gdal.WarpOptions(**vrt_dict)

        if return_vrt_dict:
            return vrt_dict
        else:
            if outfile is None:
                outfile = Handler(self.name).add_to_filename('_clipped')

            vrt_ds = gdal.Warp(outfile, self.name, options=vrt_opt)

            if return_vrt:
                return vrt_ds
            else:
                vrt_ds = None
                return Raster(outfile)

    def reproject(self,
                  outfile=None,
                  out_epsg=None,
                  out_wkt=None,
                  out_proj4=None,
                  out_spref=None,
                  output_res=None,
                  out_datatype=gdal.GDT_Float32,
                  resampling=None,
                  output_bounds=None,
                  out_format='GTiff',
                  out_nodatavalue=None,
                  verbose=False,
                  cutline_file=None,
                  cutline_blend=0,
                  return_datasource=False,
                  **creation_options):
        """
        Method to reproject raster object
        :param outfile: output file name
        :param out_epsg: EPSG code for output spatial reference
        :param out_wkt: WKT representation of output spatial reference
        :param out_proj4: PROJ4 representation of output spatial reference
        :param out_spref: Output spatial reference object
        :param output_res: output spatial resolution (xRes, yRes)
        :param out_datatype: output type (gdal.GDT_Byte, etc...)
        :param resampling: near, bilinear, cubic, cubicspline,
                           lanczos, average, mode, max, min, med, q1, q3
        :param out_nodatavalue: output no-data value to replace input no-data value
        :param output_bounds: output bounds as (minX, minY, maxX, maxY) in target SRS
        :param out_format: output format ("GTiff", etc...)
        :param verbose: If the steps should be displayed
        :param cutline_file: Shapefile, etc. to clip raster
        :param cutline_blend: blend distance in pixels
        :param return_datasource: If datasource object should be returned instead of raster
        :param creation_options: Creation options to be used while writing the raster
                                (example for geotiff: 'compress=lzw' , 'bigtiff=yes' )
        :return: VRT object or None (if output file is also specified)

        All the other valid warp options can be found at
        https://gdal.org/python/osgeo.gdal-module.html#WarpOptions
        """

        warp_dict = dict()

        if output_bounds is not None:
            warp_dict['outputBounds'] = output_bounds

        if output_res is not None:
            warp_dict['xRes'] = output_res[0]
            warp_dict['yRes'] = output_res[1]

        if out_nodatavalue is not None:
            warp_dict['dstNodata'] = out_nodatavalue
        else:
            warp_dict['dstNodata'] = self.nodatavalue

        warp_dict['srcNodata'] = self.nodatavalue

        if resampling is not None:
            warp_dict['resampleAlg'] = resampling
        else:
            warp_dict['resampleAlg'] = 'near'

        if verbose:
            Opt.cprint('Outfile: {}'.format(outfile))

        if out_spref is not None:
            spref = out_spref
        else:
            spref = osr.SpatialReference()

            if out_epsg is not None:
                res = spref.ImportFromEPSG(out_epsg)
            elif out_wkt is not None:
                res = spref.ImportFromWkt(out_wkt)
            elif out_proj4 is not None:
                res = spref.ImportFromProj4(out_proj4)
            else:
                res = spref.ImportFromWkt(self.crs_string)

        warp_dict['srcSRS'] = self.crs_string
        warp_dict['dstSRS'] = spref.ExportToWkt()

        warp_dict['outputType'] = out_datatype
        warp_dict['copyMetadata'] = True
        warp_dict['format'] = out_format

        if cutline_file is not None:
            clip_opts = self.clip(cutline_file,
                                  cutline_blend,
                                  return_vrt_dict=True)

            warp_dict.update(clip_opts)

        creation_options_list = []
        if len(creation_options) > 0:
            for key, value in creation_options.items():
                creation_options_list.append('{}={}'.format(str(key).upper(),
                                                            str(value).upper()))

            warp_dict['creationOptions'] = creation_options_list

        warp_opts = gdal.WarpOptions(**warp_dict)

        if outfile is None:
            outfile = Handler(self.name).dirname + Handler().sep + '_reproject.tif'

        warp_datasource = gdal.Warp(outfile,
                                    self.name,
                                    options=warp_opts)

        if return_datasource:
            return warp_datasource
        else:
            vrt_ds = None


class MultiRaster:
    """
    Virtual raster class to manipulate GDAL virtual raster object
    """

    def __init__(self,
                 filelist=None,
                 initialize=True,
                 get_array=False):

        """
        Class constructor
        :param filelist: List of raster (.tif) files
        :param initialize: if the rasters from file list should be initialized
        :param get_array: if raster arrays should be read to memory
        """

        self.filelist = filelist
        self.rasters = list()

        if filelist is not None:

            if type(filelist).__name__ not in ('list', 'tuple'):
                filelist = [filelist]
            for filename in filelist:
                ras = Raster(filename)
                if initialize:
                    ras.initialize(get_array=get_array)
                self.rasters.append(ras)

        self.intersection = None
        self.nodatavalue = list(raster.nodatavalue for raster in self.rasters)
        self.resolutions = list((raster.transform[1], raster.transform[5]) for raster in self.rasters)

    def get_extent(self,
                   index=None,
                   intersection=True,
                   verbose=False,
                   _return=True):
        """
        Method to get intersecting bounds of the raster objects
        :param index: list of indices of raster files/objects
        :param verbose: If some of the steps should be displayed
        :param intersection: If intersection (True) should be returned or union (False) (Default: True)
        :param _return: Should the method return the bound coordinates
        :return: coordinates of intersection (minx, miny, maxx, maxy)
        """

        wkt_list = list()
        if index is not None:
            for ii in index:
                bounds = self.rasters[ii].get_bounds()
                wktstring = 'POLYGON(({}))'.format(', '.join(list(' '.join([str(x), str(y)])
                                                                  for (x, y) in bounds)))
                wkt_list.append(wktstring)
        else:
            for raster in self.rasters:
                bounds = raster.get_bounds()
                wktstring = 'POLYGON(({}))'.format(', '.join(list(' '.join([str(x), str(y)])
                                                                  for (x, y) in bounds)))
                wkt_list.append(wktstring)

        geoms = list(ogr.CreateGeometryFromWkt(wktstring) for wktstring in wkt_list)

        if intersection:
            temp_geom = geoms[0]
            for geom in geoms[1:]:
                temp_geom = temp_geom.Intersection(geom)
        else:
            temp_geom = ogr.Geometry(ogr.wkbMultiPolygon)

            for geom in geoms:
                temp_geom.AddGeometryDirectly(geom)

        temp_geom_json = temp_geom.ExportToJson()
        json_str = json.loads(temp_geom_json)

        temp_coords = []
        for level1_list in json_str['coordinates']:
            for level2_list in level1_list:
                if json_str['type'] == 'Polygon':
                    temp_coords.append(level2_list)
                elif json_str['type'] == 'MultiPolygon':
                    for level3_list in level2_list:
                        temp_coords.append(level3_list)
                else:
                    temp_coords.append(None)

        minx = min(list(coord[0] for coord in temp_coords))
        miny = min(list(coord[1] for coord in temp_coords))

        maxx = max(list(coord[0] for coord in temp_coords))
        maxy = max(list(coord[1] for coord in temp_coords))

        self.intersection = (minx, miny, maxx, maxy)

        if verbose:
            Opt.cprint('xmin: {}, ymin: {}, xmax: {}, ymax: {}'.format(*[str(elem)
                                                                         for elem in
                                                                         self.intersection]))

        if _return:
            return self.intersection

    def layerstack(self,
                   order=None,
                   verbose=False,
                   outfile=None,
                   return_vrt=True,
                   **kwargs):

        """
        Method to layerstack rasters in a given order
        :param order: order of raster layerstack
        :param verbose: If some of the steps should be printed to console
        :param outfile: Name of the output file (.tif)
        :param return_vrt: If the file should be written to disk or vrt object should be returned
        :return: None

        valid build vrt options in kwargs
        (from https://gdal.org/python/osgeo.gdal-module.html#BuildVRT):

        valid translate options in kwargs
        (from https://gdal.org/python/osgeo.gdal-module.html#TranslateOptions):
        """

        if order is None:
            order = list(range(len(self.rasters)))

        vrt_dict = dict()

        if 'output_bounds' in kwargs:
            vrt_dict['outputBounds'] = kwargs['output_bounds']
        else:
            vrt_dict['outputBounds'] = self.intersection

        output_res = min(list(np.abs(self.resolutions[i][0]) for i in order))

        if 'outputresolution' in kwargs:
            vrt_dict['xRes'] = kwargs['outputresolution'][0]
            vrt_dict['yRes'] = kwargs['outputresolution'][1]
        else:
            vrt_dict['xRes'] = output_res
            vrt_dict['yRes'] = output_res

        if 'nodatavalue' in kwargs:
            vrt_dict['srcNodata'] = kwargs['nodatavalue']
        else:
            vrt_dict['srcNodata'] = self.nodatavalue[0]

        if 'outnodatavalue' in kwargs:
            vrt_dict['VRTNodata'] = kwargs['outnodatavalue']
        else:
            vrt_dict['VRTNodata'] = self.nodatavalue[0]

        if 'resample' in kwargs:
            vrt_dict['resampleAlg'] = kwargs['resample']
        else:
            vrt_dict['resampleAlg'] = 'cubic'

        if verbose:
            Opt.cprint('Getting bounds ...')

        vrt_dict['outputBounds'] = self.get_extent(index=order)
        vrt_dict['separate'] = True
        vrt_dict['hideNodata'] = False

        if verbose:
            Opt.cprint('Files: \n{}'.format('\n'.join(list(self.filelist[i] for i in order))))

        _vrt_opt_ = gdal.BuildVRTOptions(**vrt_dict)

        if outfile is None:
            vrtfile = Handler(self.filelist[0]).dirname + Handler().sep + 'layerstack1.vrt'
            outfile = Handler(self.filelist[0]).dirname + Handler().sep + 'layerstack1.tif'
        else:
            vrtfile = outfile.replace('.tif', '.vrt')

        _vrt_ = gdal.BuildVRT(vrtfile,
                              list(self.filelist[i] for i in order),
                              options=_vrt_opt_)

        if not return_vrt:

            vrt_dict['creationOptions'] = []

            if 'compress' in kwargs:
                vrt_dict['creationOptions'].append('COMPRESS={}'.format(str(kwargs['compress']).upper()))

            if 'bigtiff' in kwargs:
                vrt_dict['creationOptions'].append('BIGTIFF={}'.format(str(kwargs['bigtiff']).upper()))

            if verbose:
                Opt.cprint('Writing layer stack file : {} ...'.format(outfile))

            nodata = vrt_dict.pop('VRTNodata')
            _ = vrt_dict.pop('srcNodata')
            _ = vrt_dict.pop('hideNodata')
            _ = vrt_dict.pop('separate')

            if 'outputBounds' in vrt_dict:
                _ = vrt_dict.pop('xRes')
                _ = vrt_dict.pop('yRes')

            vrt_dict['noData'] = nodata

            trns_opts = gdal.TranslateOptions(**vrt_dict)

            gdal.Translate(outfile,
                           _vrt_,
                           options=trns_opts)
            _vrt_ = None

            if verbose:
                Opt.cprint('Done!')
        else:
            return _vrt_

    def composite(self,
                  layer_indices=None,
                  verbose=False,
                  outfile=None,
                  composite_type='mean',
                  tile_size=1024,
                  write_raster=False,
                  **kwargs):
        """
        Method to calculate raster composite in a given order
        :param layer_indices: list of layer indices
        :param verbose: If some of the steps should be printed to console
        :param outfile: Name of the output file (.tif)
        :param tile_size: Size of internal tile
        :param write_raster: If the raster file should be written or a raster object be returned
        :param composite_type: mean, median, pctl_xxx (eg: pctl_5, pctl_99, pctl_100, etc.),
        :return: None
        """

        if layer_indices is None:
            layer_indices = list(range(len(self.rasters)))  # list of layer indices
            t_order = list(range(1, len(self.rasters) + 1))  # list of bands to include in raster tiles
        else:
            t_order = list(elem + 1 for elem in layer_indices)  # list of bands to include in raster tiles

        # layer stack vrt
        _ls_vrt_ = self.layerstack(order=layer_indices,
                                   verbose=verbose,
                                   return_vrt=True,
                                   **kwargs)

        # raster object from vrt
        lras = Raster('tmp_layerstack')
        lras.datasource = _ls_vrt_
        lras.initialize()

        if 'bound_coords' in kwargs:
            if 'coords_type' in kwargs:
                lras.make_tile_grid(tile_size,
                                    tile_size,
                                    bound_coords=kwargs['bound_coords'],
                                    coords_type=kwargs['coords_type'])
            else:
                lras.make_tile_grid(tile_size,
                                    tile_size,
                                    bound_coords=kwargs['bound_coords'],
                                    coords_type='crs')
        else:
            lras.make_tile_grid(tile_size,
                                tile_size)

        if verbose:
            Opt.cprint(lras)

        # make numpy array to hold the final result
        out_arr = np.zeros((lras.shape[1], lras.shape[2]),
                           dtype=gdal_array.GDALTypeCodeToNumericTypeCode(lras.dtype))

        # loop through raster tiles
        count = 0
        for tie_pt, tile_arr in lras.get_next_tile(bands=t_order):

            if verbose:
                Opt.cprint(lras.tile_grid[count]['block_coords'])

            _x, _y, _cols, _rows = lras.tile_grid[count]['block_coords']

            if composite_type == 'mean':
                temp_arr = np.apply_along_axis(lambda x: np.mean(x[x != lras.nodatavalue]), 0, tile_arr)
            elif composite_type == 'median':
                temp_arr = np.apply_along_axis(lambda x: np.median(x[x != lras.nodatavalue]), 0, tile_arr)
            elif composite_type == 'min':
                temp_arr = np.apply_along_axis(lambda x: np.min(x[x != lras.nodatavalue]), 0, tile_arr)
            elif composite_type == 'max':
                temp_arr = np.apply_along_axis(lambda x: np.max(x[x != lras.nodatavalue]), 0, tile_arr)
            elif 'pctl' in composite_type:
                pctl = int(composite_type.split('_')[1])
                temp_arr = np.apply_along_axis(lambda x: np.percentile(x[x != lras.nodatavalue], pctl), 0, tile_arr)
            else:
                raise ImageProcessingError('Unknown composite option')

            # update output array with tile composite
            out_arr[_y: (_y+_rows), _x: (_x+_cols)] = temp_arr
            count += 1

        # write array to raster
        lras.array = out_arr

        if write_raster:
            # write raster
            lras.write_to_file(outfile)
            if verbose:
                Opt.cprint('Written {}'.format(outfile))
        else:
            return lras

    def mosaic(self,
               order=None,
               verbose=False,
               outfile=None,
               nodata_values=None,
               blend_pixels=10,
               blend_cutline=None,
               add_overviews=False,
               return_datasource=False,
               **kwargs):
        """
        Method to mosaic rasters in a given order
        This method uses all bands of the given rasters
        :param order: order of raster layerstack
        :param verbose: If some of the steps should be printed to console
        :param outfile: Name of the output file (.tif)
        :param nodata_values: Value or tuple (or list) of values used as
                             nodata bands for each image to be mosaicked
        :param blend_pixels: width of pixels to blend around the cutline or
                     raster boundary for multiple rasters (default: 10)
        :param blend_cutline: vector file (shapefile) in memory or on disk
                              to use for blending
        :param add_overviews: If overviews should be added to the resulting image
        :param return_datasource: If datasource of the mosaic output should be returned
        :param kwargs: Other options
                       valid warp options in kwargs :
                       https://gdal.org/python/osgeo.gdal-module.html#WarpOptions
        :return: None
        todo: 1) band selection: Use VRT to get bands that will be used at runtime
              2) blending options: image bounds buffered at intersection with other image
              using:
                from scipy.ndimage.morphology import binary_erosion
                from scipy.spatial.distance import cdist
                def dist_from_edge(img):
                    I = binary_erosion(img) # Interior mask
                    C = img - I             # Contour mask
                    out = C.astype(int)     # Setup o/p and assign cityblock distances
                    out[I] = cdist(np.argwhere(C), np.argwhere(I), 'cityblock').min(0) + 1
                    return out
                    OR
                    import numpy as np
                    from scipy.ndimage import distance_transform_cdt
                    def distance_from_edge(x):
                        x = np.pad(x, 1, mode='constant')
                        dist = distance_transform_cdt(x, metric='taxicab')
                        return dist[1:-1, 1:-1]
                    OR
                    import numpy as np
                    from scipy.spatial.distance import cdist
                    def feature_dist(input):
                        # Takes a labeled array as returned by scipy.ndimage.label and
                        # returns an intra-feature distance matrix.
                        I, J = np.nonzero(input)
                        labels = input[I,J]
                        coords = np.column_stack((I,J))
                        sorter = np.argsort(labels)
                        labels = labels[sorter]
                        coords = coords[sorter]
                        sq_dists = cdist(coords, coords, 'sqeuclidean')
                        start_idx = np.flatnonzero(np.r_[1, np.diff(labels)])
                        nonzero_vs_feat = np.minimum.reduceat(sq_dists, start_idx, axis=1)
                        feat_vs_feat = np.minimum.reduceat(nonzero_vs_feat, start_idx, axis=0)
                        return np.sqrt(feat_vs_feat)
                        This approach requires O(N2) memory,
                        where N is the number of nonzero pixels.
                        If this is too demanding,
                        you could "de-vectorize" it along one axis (add a for-loop).
        """

        if order is None:
            order = list(range(len(self.filelist)))

        if verbose:
            Opt.cprint('Files: \n{}'.format('\n'.join(list(self.filelist[i] for i in order))))

        warp_dict = {}

        warp_dict['options'] = []

        output_res = max(list(np.abs(self.resolutions[i][0]) for i in order))

        if 'output_resolution' in kwargs:
            warp_dict['xRes'] = kwargs['output_resolution'][0]
            warp_dict['yRes'] = kwargs['output_resolution'][1]
        else:
            warp_dict['xRes'] = output_res
            warp_dict['yRes'] = output_res

        warp_dict['srcSRS'] = self.rasters[0].crs_string

        if 'out_crs_string' in kwargs:
            warp_dict['dstSRS'] = kwargs['out_crs_string']
        else:
            warp_dict['dstSRS'] = warp_dict['srcSRS']

        if nodata_values is not None:
            warp_dict['srcNodata'] = nodata_values
        else:
            warp_dict['srcNodata'] = list(ras.nodatavalue for ras in self.rasters)

        if 'outnodatavalue' in kwargs:
            warp_dict['dstNodata'] = kwargs['outnodatavalue']
        else:
            warp_dict['dstNodata'] = self.nodatavalue[0]

        if 'resample' in kwargs:
            warp_dict['resampleAlg'] = kwargs['resample']
        else:
            warp_dict['resampleAlg'] = 'bilinear'

        if verbose:
            Opt.cprint('Getting bounds ...')

        warp_dict['outputBounds'] = self.get_extent(index=order,
                                                    verbose=verbose,
                                                    intersection=False)
        warp_dict['targetAlignedPixels'] = True
        warp_dict['outputType'] = self.rasters[0].dtype

        if blend_cutline is not None:
            blend_datasource = ogr.Open(blend_cutline)
            blend_layer = blend_datasource.GetLayerByIndex(0)
            blend_layer_name = blend_layer.GetName()

            warp_dict['cutlineDSName'] = blend_cutline
            warp_dict['cutlineLayer'] = blend_layer_name
            warp_dict['cutlineBlend'] = blend_pixels

        warp_dict['creationOptions'] = []

        if 'compress' in kwargs:
            warp_dict['creationOptions'].append('COMPRESS={}'.format(str(kwargs['compress']).upper()))

        if 'bigtiff' in kwargs:
            warp_dict['creationOptions'].append('BIGTIFF={}'.format(str(kwargs['bigtiff']).upper()))

        warp_opts = gdal.WarpOptions(**warp_dict)

        warp_datasource = gdal.Warp(outfile,
                                    self.filelist,
                                    options=warp_opts)

        if add_overviews:
            ras = Raster(outfile)
            if type(add_overviews) == list:
                ras.add_overviews(overviews=add_overviews)
            else:
                ras.add_overviews()

        if return_datasource:
            return warp_datasource

