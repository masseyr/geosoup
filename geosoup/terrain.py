from geosoup.raster import Raster, gdal
from geosoup.common import Handler


class Terrain(Raster):
    """Class to process DEM rasters"""

    def __init__(self,
                 name,
                 array=None,
                 bnames=None,
                 metadict=None,
                 dtype=None,
                 shape=None,
                 transform=None,
                 crs_string=None):

        super(Terrain, self).__init__(name,
                                      array,
                                      bnames,
                                      metadict,
                                      dtype,
                                      shape,
                                      transform,
                                      crs_string)

    def __repr__(self):
        return 'Terrain: ' + super(Terrain, self).__repr__()

    def slope(self,
              outfile=None,
              slope_format='degree',
              file_format='GTiff',
              compute_edges=True,
              band=1,
              scale=None,
              algorithm='ZevenbergenThorne',
              **creation_options):

        """
        Method to calculate slope
        :param outfile: output file name
        :param slope_format: format of the slope raster (valid options: 'degree', or 'percent')
        :param file_format: Output file format (default: 'GTiff')
        :param compute_edges: If the edges of the raster should be computed as well.
                              This can present incomplete results at the edges and resulting
                              rasters may show edge effects on mosaicking
        :param band: Band index to use (default: 0)
        :param scale: ratio of vertical to horizontal units
        :param algorithm: slope algorithm to use
                          valid options:
                             4-neighbor: 'ZevenbergenThorne'
                             8-neighbor: 'Horn'
        :param creation_options: Valid creation options examples:
                                 compress='LZW'
                                 bigtiff='yes'
        """
        if not self.init:
            self.initialize()

        if outfile is None:
            outfile = Handler(self.name).add_to_filename('_SLOPE')
            outfile = Handler(outfile).file_remove_check()

        creation_option_list = list()
        for key, value in creation_options.items():
            creation_option_list.append('{}={}'.format(str(key).upper(), str(value).upper()))

        slope_opts = gdal.DEMProcessingOptions(format=file_format,
                                               computeEdges=compute_edges,
                                               alg=algorithm,
                                               slopeFormat=slope_format,
                                               band=band,
                                               scale=scale,
                                               creationOptions=creation_option_list)

        res = gdal.DEMProcessing(outfile,
                                 self.datasource,
                                 'slope',
                                 options=slope_opts)
        res = None

    def aspect(self,
               outfile=None,
               file_format='GTiff',
               compute_edges=True,
               band=1,
               algorithm='ZevenbergenThorne',
               zero_for_flat=True,
               trigonometric=False,
               **creation_options):

        """
        Method to calculate aspect
        :param outfile: output file name
        :param file_format: Output file format (default: 'GTiff')
        :param compute_edges: If the edges of the raster should be computed as well.
                              This can present incomplete results at the edges and resulting
                              rasters may show edge effects on mosaicking
        :param band: Band index to use (default: 0)
        :param algorithm: slope algorithm to use
                          valid options:
                             4-neighbor: 'ZevenbergenThorne'
                             8-neighbor: 'Horn'

        :param zero_for_flat: whether to return 0 for flat areas with slope=0, instead of -9999.
        :param trigonometric: whether to return trigonometric angle instead of azimuth.
                             Here 0deg will mean East, 90deg North, 180deg West, 270deg South.
        :param creation_options: Valid creation options examples:
                                 compress='LZW'
                                 bigtiff='yes'
        """
        if not self.init:
            self.initialize()

        if outfile is None:
            outfile = Handler(self.name).add_to_filename('_ASPECT')
            outfile = Handler(outfile).file_remove_check()

        creation_option_list = list()
        for key, value in creation_options.items():
            creation_option_list.append('{}={}'.format(str(key), str(value)))

        aspect_opts = gdal.DEMProcessingOptions(format=file_format,
                                                computeEdges=compute_edges,
                                                creationOptions=creation_option_list,
                                                alg=algorithm,
                                                band=band,
                                                zeroForFlat=zero_for_flat,
                                                trigonometric=trigonometric)

        res = gdal.DEMProcessing(outfile,
                                 self.datasource,
                                 'aspect',
                                 options=aspect_opts)
        res = None

    def tpi(self,
            outfile=None,
            file_format='GTiff',
            compute_edges=True,
            band=1,
            **creation_options):

        """
        Method to calculate topographic position index
        :param outfile: output file name
        :param file_format: Output file format (default: 'GTiff')
        :param compute_edges: If the edges of the raster should be computed as well.
                              This can present incomplete results at the edges and resulting
                              rasters may show edge effects on mosaicking
        :param band: Band index to use (default: 0)
        :param creation_options: Valid creation options examples:
                                 compress='LZW'
                                 bigtiff='yes'
        """
        if not self.init:
            self.initialize()

        if outfile is None:
            outfile = Handler(self.name).add_to_filename('_TPI')
            outfile = Handler(outfile).file_remove_check()

        creation_option_list = list()
        for key, value in creation_options.items():
            creation_option_list.append('{}={}'.format(str(key), str(value)))

        tpi_opts = gdal.DEMProcessingOptions(format=file_format,
                                             computeEdges=compute_edges,
                                             creationOptions=creation_option_list,
                                             band=band)

        res = gdal.DEMProcessing(outfile,
                                 self.datasource,
                                 'TPI',
                                 options=tpi_opts)
        res = None

    def tri(self,
            outfile=None,
            file_format='GTiff',
            compute_edges=True,
            band=1,
            **creation_options):

        """
        Method to calculate topographic roughness index
        :param outfile: output file name
        :param file_format: Output file format (default: 'GTiff')
        :param compute_edges: If the edges of the raster should be computed as well.
                              This can present incomplete results at the edges and resulting
                              rasters may show edge effects on mosaicking
        :param band: Band index to use (default: 0)
        :param creation_options: Valid creation options examples:
                                 compress='LZW'
                                 bigtiff='yes'
        """
        if not self.init:
            self.initialize()

        if outfile is None:
            outfile = Handler(self.name).add_to_filename('_TRI')
            outfile = Handler(outfile).file_remove_check()

        creation_option_list = list()
        for key, value in creation_options.items():
            creation_option_list.append('{}={}'.format(str(key), str(value)))

        tpi_opts = gdal.DEMProcessingOptions(format=file_format,
                                             computeEdges=compute_edges,
                                             creationOptions=creation_option_list,
                                             band=band)

        res = gdal.DEMProcessing(outfile,
                                 self.datasource,
                                 'TRI',
                                 options=tpi_opts)
        res = None

    def roughness(self,
                  outfile=None,
                  file_format='GTiff',
                  compute_edges=True,
                  band=1,
                  **creation_options):

        """
        Method to calculate DEM roughness
        :param outfile: output file name
        :param file_format: Output file format (default: 'GTiff')
        :param compute_edges: If the edges of the raster should be computed as well.
                              This can present incomplete results at the edges and resulting
                              rasters may show edge effects on mosaicking
        :param band: Band index to use (default: 0)
        :param creation_options: Valid creation options examples:
                                 compress='LZW'
                                 bigtiff='yes'
        """
        if not self.init:
            self.initialize()

        if outfile is None:
            outfile = Handler(self.name).add_to_filename('_ROUGHNESS')
            outfile = Handler(outfile).file_remove_check()

        creation_option_list = list()
        for key, value in creation_options.items():
            creation_option_list.append('{}={}'.format(str(key), str(value)))

        tpi_opts = gdal.DEMProcessingOptions(format=file_format,
                                             computeEdges=compute_edges,
                                             creationOptions=creation_option_list,
                                             band=band)

        res = gdal.DEMProcessing(outfile,
                                 self.datasource,
                                 'Roughness',
                                 options=tpi_opts)
        res = None

    def hillshade(self,
                  outfile=None,
                  file_format='GTiff',
                  compute_edges=True,
                  band=1,
                  scale=None,
                  algorithm='ZevenbergenThorne',
                  z_factor=1,
                  azimuth=315,
                  altitude=90,
                  combined=False,
                  multi_directional=False,
                  **creation_options):

        """
        Method to calculate DEM hillshade raster
        :param outfile: output file name
        :param file_format: Output file format (default: 'GTiff')
        :param compute_edges: If the edges of the raster should be computed as well.
                              This can present incomplete results at the edges and resulting
                              rasters may show edge effects on mosaicking
        :param band: Band index to use (default: 0)
        :param scale: ratio of vertical to horizontal units
        :param algorithm: slope algorithm to use
                          valid options:
                             4-neighbor: 'ZevenbergenThorne'
                             8-neighbor: 'Horn'

        :param z_factor: vertical exaggeration used to pre-multiply the elevations. (default: 1)
        :param azimuth:  azimuth of the light, in degrees. (default: 315)
                         0 if it comes from the top of the raster, 90 from the east and so on.
                         The default value, 315, should rarely be changed
                         as it is the value generally used to generate shaded maps.

        :param altitude: altitude of the light, in degrees. (default: 90)
                         90 if the light comes from above the DEM, 0 if it is raking light.
        :param combined:  whether to compute combined shading,
                         a combination of slope and oblique shading. (Default: False)
        :param multi_directional: whether to compute multi-directional shading (Default: False)

        :param creation_options: Valid creation options examples:
                                 compress='LZW'
                                 bigtiff='yes'
        """
        if not self.init:
            self.initialize()

        if outfile is None:
            outfile = Handler(self.name).add_to_filename('_HILLSHADE')
            outfile = Handler(outfile).file_remove_check()

        creation_option_list = list()
        for key, value in creation_options.items():
            creation_option_list.append('{}={}'.format(str(key), str(value)))

        tpi_opts = gdal.DEMProcessingOptions(format=file_format,
                                             computeEdges=compute_edges,
                                             creationOptions=creation_option_list,
                                             band=band,
                                             scale=scale,
                                             alg=algorithm,
                                             zFactor=z_factor,
                                             azimuth=azimuth,
                                             altitude=altitude,
                                             combined=combined,
                                             multiDirectional=multi_directional)

        res = gdal.DEMProcessing(outfile,
                                 self.datasource,
                                 'hillshade',
                                 options=tpi_opts)
        res = None

