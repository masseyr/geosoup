from osgeo import ogr, gdal

"""
This module houses definitions
"""

__all__ = ['OGR_FIELD_DEF',
           'OGR_FIELD_DEF_INV',
           'OGR_GEOM_DEF',
           'OGR_TYPE_DEF',
           'GDAL_FIELD_DEF',
           'GDAL_FIELD_DEF_INV']


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

GDAL_FIELD_DEF = {
    'byte': gdal.GDT_Byte,
    'int': gdal.GDT_Int16,
    'long': gdal.GDT_Int32,
    'float': gdal.GDT_Float32,
    'double': gdal.GDT_Float64,
    'uint': gdal.GDT_UInt16,
    'ulong': gdal.GDT_UInt32,
}


OGR_FIELD_DEF_INV = dict(list((v, k) for k, v in OGR_FIELD_DEF.items()))


GDAL_FIELD_DEF_INV = dict(list((v, k) for k, v in GDAL_FIELD_DEF.items()))


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


OGR_GEOM_DEF = {
                1: 'point',
                2: 'line',
                3: 'polygon',
                4: 'multipoint',
                5: 'multilinestring',
                6: 'multipolygon',
                0: 'geometry',
                100: 'no geometry',
}
