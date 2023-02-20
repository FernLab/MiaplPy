############################################################
# Program is part of MiaplPy                                #
# Author:  Sara Mirzaee                                    #
############################################################
# class used for data loading from slc stack to MiaplPy timeseries
# Recommend import:
#     from miaplpy.objects.slcStack import slcStackDict


import os
import time
import warnings

import h5py
import numpy as np
from datetime import datetime
from osgeo import gdal

from miaplpy.objects.arg_parser import MiaplPyParser

try:
    from skimage.transform import resize
except ImportError:
    raise ImportError('Could not import skimage!')

from mintpy.objects import (DSET_UNIT_DICT, timeseries, geometry, ifgramStack, HDFEOS, giantTimeseries,
                            giantIfgramStack)

from mintpy.utils import readfile, ptime, attribute as attr
from miaplpy.objects.utils import read_attribute, read_inps2dict, prepare_metadata, read_subset_box, \
    read_inps_dict2slc_stack_dict_object, read_binary_file
from miaplpy.objects.utils import read_binary_file

BOOL_ZERO = np.bool_(0)
INT_ZERO = np.int16(0)
FLOAT_ZERO = np.float32(0.0)
CPX_ZERO = np.complex64(0.0)

dataType = np.complex64

slcDatasetNames = ['slc']
DSET_UNIT_DICT['slc'] = 'i'
gdal.SetCacheMax(2**30)
########################################################################################

dataTypeDict = {
    'byte': 'int8',
    'float': 'float32',
    'double': 'float64',
    'cfloat': 'complex64',
}

class slcStackDict:
    '''
    slcStack object for a set of coregistered SLCs from the same platform and track.

    Example:
        from miaplpy.objects.insarobj import slcStackDict
        pairsDict = {('20160524','20160530'):slcObj1,
                     ('20160524','20160605'):slcObj2,
                     ('20160524','20160611'):slcObj3,
                     ('20160530','20160605'):slcObj4,
                     ...
                     }
        stackObj = slcStackDict(pairsDict=pairsDict)
        stackObj.write2hdf5(outputFile='slcStack.h5', box=(200,500,300,600))
    '''

    def __init__(self, name='slc', pairsDict=None):
        self.name = name
        #self.pairsDict = pairsDict
        self.pairsDict = pairsDict

    def get_size(self, box=None, xstep=1, ystep=1):
        self.numSlc = len(self.pairsDict)
        slcObj = [v for v in self.pairsDict.values()][0]
        self.length, slcObj.width = slcObj.get_size()
        if box:
            self.length = box[3] - box[1]
            self.width = box[2] - box[0]
        else:
            self.length = slcObj.length
            self.width = slcObj.width

        # update due to multilook
        self.length = int(self.length // ystep)
        self.width = int(self.width // xstep)

        return self.numSlc, self.length, self.width

    def get_date_list(self):
        self.dateList = sorted([date for date in self.pairsDict.keys()])
        return self.dateList

    def get_metadata(self):
        slcObj = [v for v in self.pairsDict.values()][0]
        self.metadata = slcObj.get_metadata()
        if 'UNIT' in self.metadata.keys():
            self.metadata.pop('UNIT')
        return self.metadata

    def get_dataset_data_type(self, dsName):
        slcObj = [v for v in self.pairsDict.values()][0]
        dsFile = slcObj.datasetDict[dsName]
        metadata = read_attribute(dsFile.split('.xml')[0], metafile_ext='.rsc')
        dsDataType = dataType
        if 'DATA_TYPE' in metadata.keys():
            dsDataType = dataTypeDict[metadata['DATA_TYPE'].lower()]
        return dsDataType

    def write2hdf5(self, outputFile='slcStack.h5', access_mode='a', box=None, xstep=1,
                            ystep=1, compression=None, extra_metadata=None):
        '''Save/write an slcStackDict object into an HDF5 file with the structure below:

        /                  Root level
        Attributes         Dictionary for metadata
        /date              2D array of string  in size of (m, 2   ) in YYYYMMDD format for reference and secondary date
        /bperp             1D array of float32 in size of (m,     ) in meter.

        Parameters: outputFile : str, Name of the HDF5 file for the SLC stack
                    access_mode : str, access mode of output File, e.g. w, r+
                    box : tuple, subset range in (x0, y0, x1, y1)
                    extra_metadata : dict, extra metadata to be added into output file
        Returns:    outputFile
        '''
        self.outputFile = outputFile
        f = h5py.File(self.outputFile, access_mode)
        print('create HDF5 file {} with {} mode'.format(self.outputFile, access_mode))

        # self.pairs = sorted([pair for pair in self.pairsDict.keys()])
        # self.dsNames = list(self.pairsDict[self.pairs[0]].datasetDict.keys())
        self.dates = sorted([date for date in self.pairsDict.keys()])
        self.dsNames = list(self.pairsDict[self.dates[0]].datasetDict.keys())
        self.dsNames = [i for i in slcDatasetNames if i in self.dsNames]
        maxDigit = max([len(i) for i in self.dsNames])
        self.get_size(box=box, xstep=xstep, ystep=ystep)

        self.bperp = np.zeros(self.numSlc)

        ###############################

        # 3D datasets containing slc.
        for dsName in self.dsNames:
            dsShape = (self.numSlc, self.length, self.width)
            dsDataType = dataType
            dsCompression = compression
            if dsName in ['connectComponent']:
                dsDataType = np.int16
                dsCompression = 'lzf'

            print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
                   ' with compression = {c}').format(d=dsName,
                                                     w=maxDigit,
                                                     t=str(dsDataType),
                                                     s=dsShape,
                                                     c=dsCompression))

            if dsName in f.keys():
                ds = f[dsName]
            else:
                ds = f.create_dataset(dsName,
                                      shape=dsShape,
                                      maxshape=(None, dsShape[1], dsShape[2]),
                                      dtype=dsDataType,
                                      chunks=True,
                                      compression=dsCompression)

                prog_bar = ptime.progressBar(maxValue=self.numSlc)

                for i in range(self.numSlc):
                    slcObj = self.pairsDict[self.dates[i]]
                    # fname, metadata = slcObj.read(dsName, box=box)
                    fname, metadata = read(dsName)

                    if not box:
                        box = (0, 0, self.width, self.length)
                    dsSlc = gdal.Open(fname + '.vrt', gdal.GA_ReadOnly)
                    ds[i, :, :] = dsSlc.GetRasterBand(1).ReadAsArray(int(box[0]), int(box[1]), self.width, self.length)

                    self.bperp[i] = slcObj.get_perp_baseline()
                    prog_bar.update(i+1, suffix='{}'.format(self.dates[i][0]))

                prog_bar.close()
            ds.attrs['MODIFICATION_TIME'] = str(time.time())

        ###############################
        # 1D dataset containing dates of all images
        dsName = 'date'
        dsDataType = np.string_
        dsShape = (self.numSlc, 1)
        print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                          w=maxDigit,
                                                                          t=str(dsDataType),
                                                                          s=dsShape))
        data = np.array(self.dates, dtype=dsDataType)
        if not dsName in f.keys():
            f.create_dataset(dsName, data=data)

        ###############################
        # 1D dataset containing perpendicular baseline of all pairs
        dsName = 'bperp'
        dsDataType = np.float32
        dsShape = (self.numSlc,)
        print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                          w=maxDigit,
                                                                          t=str(dsDataType),
                                                                          s=dsShape))
        data = np.array(self.bperp, dtype=dsDataType)
        if not dsName in f.keys():
            f.create_dataset(dsName, data=data)
        

        ###############################
        # Attributes
        self.get_metadata()
        if extra_metadata:
            self.metadata.update(extra_metadata)
            #print('add extra metadata: {}'.format(extra_metadata))
        self.metadata = attr.update_attribute4subset(self.metadata, box)

        # update metadata due to multilook
        if xstep * ystep > 1:
            self.metadata = attr.update_attribute4multilook(self.metadata, ystep, xstep)

        self.metadata['FILE_TYPE'] = 'timeseries' #'slc'
        for key, value in self.metadata.items():
            f.attrs[key] = value

        f.close()
        print('Finished writing to {}'.format(self.outputFile))
        return self.outputFile



################################ slcStack class begin ################################
FILE_STRUCTURE_SLCs = """
/                Root level
Attributes       Dictionary for metadata
/slc             3D array of float32 in size of (n, l, w) in meter.
/date            1D array of string  in size of (n,     ) in YYYYMMDD format
/bperp           1D array of float32 in size of (n,     ) in meter. (optional)
"""

class slcStack:
    """
    Time-series object for displacement of a set of SAR images from the same platform and track.
    It contains three datasets in root level: date, bperp and SLCs.
    """

    def __init__(self, file=None):
        self.file = file
        self.name = 'slc'
        self.file_structure = FILE_STRUCTURE_SLCs

    def close(self, print_msg=True):
        try:
            self.f.close()
            if print_msg:
                print('close slcStack file: {}'.format(os.path.basename(self.file)))
        except:
            pass
        return None

    def open_hdf5(self, mode='a'):
        print('open {} in {} mode'.format(self.file, mode))
        self.f = h5py.File(self.file, mode)
        return self.f

    def open(self, print_msg=True):
        if print_msg:
            print('open {} file: {}'.format(self.name, os.path.basename(self.file)))
        self.get_metadata()
        self.get_size()
        self.get_date_list()
        self.numPixel = self.length * self.width

        with h5py.File(self.file, 'r') as f:
            try:
                self.pbase = f['bperp'][:]
                self.pbase -= self.pbase[self.refIndex]
            except:
                self.pbase = None

        self.times = np.array([datetime(*time.strptime(i, "%Y%m%d")[0:5]) for i in self.dateList])
        self.tbase = np.array([i.days for i in self.times - self.times[self.refIndex]],
                              dtype=np.float32)
        # list of float for year, 2014.95
        self.yearList = [i.year + (i.timetuple().tm_yday-1)/365.25 for i in self.times]
        self.sliceList = ['{}-{}'.format(self.name, i) for i in self.dateList]
        return None

    def get_metadata(self):
        with h5py.File(self.file, 'r') as f:
            self.metadata = dict(f.attrs)
            dates = f['date'][:]
        for key, value in self.metadata.items():
            try:
                self.metadata[key] = value.decode('utf8')
            except:
                self.metadata[key] = value

        # ref_date/index
        dateList = [i.decode('utf8') for i in dates]
        if 'REF_DATE' not in self.metadata.keys():
            self.metadata['REF_DATE'] = dateList[0]
        self.refIndex = dateList.index(self.metadata['REF_DATE'])
        self.metadata['START_DATE'] = dateList[0]
        self.metadata['END_DATE'] = dateList[-1]
        return self.metadata

    def get_size(self,xstep=1, ystep=1):
        with h5py.File(self.file, 'r') as f:
            self.numDate, self.length, self.width = f[self.name].shape

        # update due to multilook
        self.length = self.length // ystep
        self.width = self.width // xstep
        return self.numDate, self.length, self.width

    def get_date_list(self):
        with h5py.File(self.file, 'r') as f:
            self.dateList = [i.decode('utf8') for i in f['date'][:]]
        return self.dateList

    def read(self, datasetName=None, box=None, print_msg=True):
        """Read dataset from slc file
        Parameters: self : slcStack object
                    datasetName : (list of) string in YYYYMMDD format
                    box : tuple of 4 int, indicating x0,y0,x1,y1 of range
        Returns:    data : 2D or 3D dataset
        Examples:   from miaplpy.objects import slcStack
                    tsobj = slcStack('slcStack.h5')
                    data = tsobj.read(datasetName='20161020')
                    data = tsobj.read(datasetName='20161020', box=(100,300,500,800))
                    data = tsobj.read(datasetName=['20161020','20161026','20161101'])
                    data = tsobj.read(box=(100,300,500,800))
        """
        if print_msg:
            print('reading box {} from file: {} ...'.format(box, self.file))
        self.open(print_msg=False)

        # convert input datasetName into list of dates
        if not datasetName or datasetName == 'slc':
            datasetName = []
        elif isinstance(datasetName, str):
            datasetName = [datasetName]
        datasetName = [i.replace('slc', '').replace('-', '') for i in datasetName]

        with h5py.File(self.file, 'r') as f:
            ds = f[self.name]
            if isinstance(ds, h5py.Group):  # support for old mintpy files
                ds = ds[self.name]

            # Get dateFlag - mark in time/1st dimension
            dateFlag = np.zeros((self.numDate), dtype=np.bool_)
            if not datasetName:
                dateFlag[:] = True
            else:
                for e in datasetName:
                    dateFlag[self.dateList.index(e)] = True

            # Get Index in space/2_3 dimension
            if box is None:
                box = [0, 0, self.width, self.length]

            data = ds[np.where(dateFlag==True)[0], box[1]:box[3], box[0]:box[2]]
            data = np.squeeze(data)
        return data

    def layout_hdf5(self, dsNameDict, metadata, compression=None):
        print('-'*50)
        print('create HDF5 file {} with w mode'.format(self.file))
        f = h5py.File(self.file, "w")

        for key in dsNameDict.keys():
            print("create dataset: {d:<25} of {t:<25} in size of {s}".format(
                d=key,
                t=str(dsNameDict[key][0]),
                s=dsNameDict[key][1]))

            f.create_dataset(key,
                             shape=dsNameDict[key][1],
                             dtype=dsNameDict[key][0],
                             chunks=True,
                             compression=compression)

        # write attributes
        metadata = dict(metadata)
        metadata['FILE_TYPE'] = 'timeseries' #self.name
        for key in metadata.keys():
            f.attrs[key] = metadata[key]

        print('close HDF5 file {}'.format(self.file))
        f.close()
        return self.file

    def write2hdf5_block(self, data, datasetName, block=None, mode='a'):
        """Write data to existing HDF5 dataset in disk block by block.
        Parameters: data : np.ndarray 1/2/3D matrix
                    datasetName : str, dataset name
                    block : list of 2/4/6 int, for
                        [zStart, zEnd,
                         yStart, yEnd,
                         xStart, xEnd]
                    mode : str, open mode
        Returns: self.file
        """
        if block is None:
            # data shape
            if isinstance(data, list):
                shape=(len(data),)
            else:
                shape = data.shape

            if len(shape) ==1:
                block = [0, shape[0]]
            elif len(shape) == 2:
                block = [0, shape[0],
                         0, shape[1]]
            elif len(shape) == 3:
                block = [0, shape[0],
                         0, shape[1],
                         0, shape[2]]

        print('open {} in {} mode'.format(self.file, mode))
        f = h5py.File(self.file, mode)

        print("writing dataset /{:<25} block: {}".format(datasetName, block))
        if len(block) == 6:
            f[datasetName][block[0]:block[1],
                           block[2]:block[3],
                           block[4]:block[5]] = data

        elif len(block) == 4:
            f[datasetName][block[0]:block[1],
                           block[2]:block[3]] = data

        elif len(block) == 2:
            f[datasetName][block[0]:block[1]] = data

        f.close()
        print('close HDF5 file {}'.format(self.file))
        return self.file

    def write2hdf5(self, data, outFile=None, dates=None, bperp=None, metadata=None, refFile=None, compression=None):
        """
        Parameters: data  : 3D array of float32
                    dates : 1D array/list of string in YYYYMMDD format
                    bperp : 1D array/list of float32 (optional)
                    metadata : dict
                    outFile : string
                    refFile : string
                    compression : string or None
        Returns: outFile : string
        Examples:
            from mintpy.objects import slcStack

            ##Generate a new slcStack file
            tsobj = slcStack('slcStack.h5')
            tsobj.write(data, dates=dateList, bperp=bperp, metadata=atr)

            ##Generate a slcStack with same attributes and same date/bperp info
            tsobj = slcStack('slcStack_modified.h5')
            tsobj.write(data, refFile='slcStack.h5')
        """

        if not outFile:
            outFile = self.file
        if refFile:
            refobj = slcStack(refFile)
            refobj.open(print_msg=False)
            if metadata is None:
                metadata = refobj.metadata
            if dates is None:
                dates = refobj.dateList
            if bperp is None:
                bperp = refobj.pbase
            # get ref file compression type if input compression is None
            if compression is None:
                with h5py.File(refFile, 'r') as rf:
                    compression = rf['slc'].compression
            refobj.close(print_msg=False)
        data = np.array(data, dtype='c16')
        dates = np.array(dates, dtype=np.string_)
        bperp = np.array(bperp, dtype=np.float32)
        metadata = dict(metadata)
        metadata['FILE_TYPE'] = 'timeseries' #self.name

        # 3D dataset - slcStack
        print('create slcStack HDF5 file: {} with w mode'.format(outFile))
        f = h5py.File(outFile, 'w')
        print(('create dataset /slcStack of {t:<10} in size of {s} '
               'with compression={c}').format(t=str(data.dtype),
                                              s=data.shape,
                                              c=compression))
        f.create_dataset('slc', data=data, chunks=True, compression=compression)

        # 1D dataset - date / bperp
        print('create dataset /date      of {:<10} in size of {}'.format(str(dates.dtype), dates.shape))
        f.create_dataset('date', data=dates)

        if bperp.shape != ():
            print('create dataset /bperp      of {:<10} in size of {}'.format(str(bperp.dtype), bperp.shape))
            f.create_dataset('bperp', data=bperp)

        # Attributes
        for key, value in metadata.items():
            f.attrs[key] = str(value)

        f.close()
        print('finished writing to {}'.format(outFile))
        return outFile

################################ slcStack class end ##################################


class slcDict:
    """
    SLC object. It includes dataset name (family) of {'slc'}

    Example:
        from mintpy.objects.insarobj import slcDict
        datasetDict = {'slc'     :'$PROJECT_DIR/merged/SLC/20151220/20151220.slc.full',
                      }
        slcObj = slcDict(dates=('20160524','20160530'), datasetDict=datasetDict)
        data, atr = slcObj.read('slc')
    """

    def __init__(self, name='slc', dates=None, datasetDict={}, metadata=None):
        self.name = name
        self.date = dates
        self.datasetDict = datasetDict

        self.platform = None
        self.track = None
        self.processor = None
        # platform, track and processor can get values from metadat if they exist
        if metadata is not None:
            for key, value in metadata.items():
                setattr(self, key, value)

    def read(self, family, box=None, datasetName=None):
        fname = self.datasetDict[family].split('.xml')[0]

        # metadata
        dsname4atr = None  # used to determine UNIT
        if isinstance(datasetName, list):
            dsname4atr = datasetName[0].split('-')[0]
        elif isinstance(datasetName, str):
            dsname4atr = datasetName.split('-')[0]
        atr = read_attribute(fname, datasetName=dsname4atr, metafile_ext='.rsc')


        # Read Data
        fext = os.path.splitext(os.path.basename(fname))[1].lower()
        if fext in ['.h5', '.he5']:
            # box
            length, width = int(atr['LENGTH']), int(atr['WIDTH'])
            if not box:
                box = (0, 0, width, length)
            data = readfile.read_hdf5_file(fname, datasetName=datasetName, box=box)
            return data
        else:
            # data, metadata = read_binary_file(fname, datasetName=datasetName, box=box)
            metadata = read_binary_file(fname, datasetName=datasetName, attributes_only=True)
            return fname, metadata

    def get_size(self, family='slc'):
        self.file = self.datasetDict[family].split('.xml')[0]
        metadata = read_attribute(self.file, metafile_ext='.rsc')
        self.length = int(metadata['LENGTH'])
        self.width = int(metadata['WIDTH'])
        return self.length, self.width

    def get_perp_baseline(self, family='slc'):
        self.file = self.datasetDict[family].split('.xml')[0]
        metadata = read_attribute(self.file, metafile_ext='.rsc')
        self.bperp_top = float(metadata['P_BASELINE_TOP_HDR'])
        self.bperp_bottom = float(metadata['P_BASELINE_BOTTOM_HDR'])
        self.bperp = (self.bperp_top + self.bperp_bottom) / 2.0
        return self.bperp

    def get_metadata(self, family='slc'):
        self.file = self.datasetDict[family].split('.xml')[0]
        self.metadata = read_attribute(self.file, metafile_ext='.rsc')
        self.length = int(self.metadata['LENGTH'])
        self.width = int(self.metadata['WIDTH'])

        # if self.processor is None:
        #    ext = self.file.split('.')[-1]
        #    if 'PROCESSOR' in self.metadata.keys():
        #        self.processor = self.metadata['PROCESSOR']
        #    elif os.path.exists(self.file+'.xml'):
        #        self.processor = 'isce'
        #    elif os.path.exists(self.file+'.rsc'):
        #        self.processor = 'roipac'
        #    elif os.path.exists(self.file+'.par'):
        #        self.processor = 'gamma'
        #    elif ext == 'grd':
        #        self.processor = 'gmtsar'
        #    #what for DORIS/SNAP
        #    else:
        #        self.processor = 'isce'
        #self.metadata['PROCESSOR'] = self.processor

        if self.track:
            self.metadata['TRACK'] = self.track

        if self.platform:
            self.metadata['PLATFORM'] = self.platform

        return self.metadata


def read_initial_info(work_dir, templateFile):
    from miaplpy.objects.slcStack import slcStack
    #import miaplpy.workflow

    slc_file = os.path.join(work_dir, 'inputs/slcStack.h5')

    if os.path.exists(slc_file):
        slcObj = slcStack(slc_file)
        slcObj.open(print_msg=False)
        date_list = slcObj.get_date_list()
        metadata = slcObj.get_metadata()
        num_pixels = int(metadata['LENGTH']) * int(metadata['WIDTH'])
    else:
        scp_args = '--template {}'.format(templateFile)
        scp_args += ' --project_dir {} --work_dir {}'.format(os.path.dirname(work_dir), work_dir)

        Parser_LoadSlc = MiaplPyParser(scp_args.split(), script='load_slc')
        inps_loadSlc = Parser_LoadSlc.parse()
        iDict = read_inps2dict(inps_loadSlc)
        prepare_metadata(iDict)
        metadata = read_subset_box(iDict)
        box = metadata['box']
        num_pixels = (box[2] - box[0]) * (box[3] - box[1])
        stackObj = read_inps_dict2slc_stack_dict_object(iDict)
        date_list = stackObj.get_date_list()

    return date_list, num_pixels, metadata


def get_slice_list(fname):
    """Get list of 2D slice existed in file (for display)"""
    fbase, fext = os.path.splitext(os.path.basename(fname))
    fext = fext.lower()
    atr = read_attribute(fname)
    k = atr['FILE_TYPE']

    global slice_list
    # HDF5 Files
    if fext in ['.h5', '.he5']:
        with h5py.File(fname, 'r') as f:
            d1_list = [i for i in f.keys() if isinstance(f[i], h5py.Dataset)]
        if k == 'timeseries' and k in d1_list:
            obj = timeseries(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k == 'slc':
            obj = slcStack(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['geometry'] and k not in d1_list:
            obj = geometry(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['ifgramStack']:
            obj = ifgramStack(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['HDFEOS']:
            obj = HDFEOS(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['giantTimeseries']:
            obj = giantTimeseries(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['giantIfgramStack']:
            obj = giantIfgramStack(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        else:
            ## Find slice by walking through the file structure
            length, width = int(atr['LENGTH']), int(atr['WIDTH'])

            def get_hdf5_2d_dataset(name, obj):
                global slice_list
                if isinstance(obj, h5py.Dataset) and obj.shape[-2:] == (length, width):
                    if obj.ndim == 2:
                        slice_list.append(name)
                    else:
                        warnings.warn('file has un-defined {}D dataset: {}'.format(obj.ndim, name))

            slice_list = []
            with h5py.File(fname, 'r') as f:
                f.visititems(get_hdf5_2d_dataset)

    # Binary Files
    else:
        if fext.lower() in ['.trans', '.utm_to_rdc']:
            slice_list = ['rangeCoord', 'azimuthCoord']
        elif fbase.startswith('los'):
            slice_list = ['incidenceAngle', 'azimuthAngle']
        elif atr.get('number_bands', '1') == '2' and 'unw' not in k:
            slice_list = ['band1', 'band2']
        else:
            slice_list = ['']
    return slice_list


def write_layout_hdf5(fname, ds_name_dict=None, metadata=None, ds_unit_dict=None, ref_file=None, compression=None, print_msg=True):
    """Create HDF5 file with defined metadata and (empty) dataset structure

    Parameters: fname        - str, HDF5 file path
                ds_name_dict - dict, dataset structure definition
                               {dname : [dtype, dshape],
                                dname : [dtype, dshape, None],
                                dname : [dtype, dshape, 1/2/3/4D np.ndarray], #for aux data
                                ...
                               }
                metadata     - dict, metadata
                ds_unit_dict - dict, dataset unit definition
                               {dname : dunit,
                                dname : dunit,
                                ...
                               }
                ref_file     - str, reference file for the data structure
                compression  - str, HDF5 compression type
    Returns:    fname        - str, HDF5 file path

    Example:    layout_hdf5('timeseries_ERA5.h5', ref_file='timeseries.h5')
                layout_hdf5('timeseries_ERA5.5h', ds_name_dict, metadata)

    # structure for ifgramStack
    ds_name_dict = {
        "date"             : [np.dtype('S8'), (num_ifgram, 2)],
        "dropIfgram"       : [np.bool_,       (num_ifgram,)],
        "bperp"            : [np.float32,     (num_ifgram,)],
        "unwrapPhase"      : [np.float32,     (num_ifgram, length, width)],
        "coherence"        : [np.float32,     (num_ifgram, length, width)],
        "connectComponent" : [np.int16,       (num_ifgram, length, width)],
    }

    # structure for geometry
    ds_name_dict = {
        "height"             : [np.float32, (length, width), None],
        "incidenceAngle"     : [np.float32, (length, width), None],
        "slantRangeDistance" : [np.float32, (length, width), None],
    }

    # structure for timeseries
    dates = np.array(date_list, np.string_)
    ds_name_dict = {
        "date"       : [np.dtype("S8"), (num_date,), dates],
        "bperp"      : [np.float32,     (num_date,), pbase],
        "timeseries" : [np.float32,     (num_date, length, width)],
    }
    """
    vprint = print if print_msg else lambda *args, **kwargs: None
    vprint('-'*50)

    # get meta from metadata and ref_file
    if metadata:
        meta = {key: value for key, value in metadata.items()}
    elif ref_file:
        with h5py.File(ref_file, 'r') as fr:
            meta = {key: value for key, value in fr.attrs.items()}
        vprint('grab metadata from ref_file: {}'.format(ref_file))
    else:
        raise ValueError('No metadata or ref_file found.')

    # check ds_name_dict
    if ds_name_dict is None:
        if not ref_file or not os.path.isfile(ref_file):
            raise FileNotFoundError('No ds_name_dict or ref_file found!')
        else:
            vprint('grab dataset structure from ref_file: {}'.format(ref_file))

        ds_name_dict = {}
        fext = os.path.splitext(ref_file)[1]
        shape2d = (int(meta['LENGTH']), int(meta['WIDTH']))

        if fext in ['.h5', '.he5']:
            # copy dset structure from HDF5 file
            with h5py.File(ref_file, 'r') as fr:
                # in case output mat size is different from the input ref file mat size
                shape2d_orig = (int(fr.attrs['LENGTH']), int(fr.attrs['WIDTH']))

                for key in fr.keys():
                    ds = fr[key]
                    if isinstance(ds, h5py.Dataset):

                        # auxliary dataset
                        if ds.shape[-2:] != shape2d_orig:
                            ds_name_dict[key] = [ds.dtype, ds.shape, ds[:]]

                        # dataset
                        else:
                            ds_shape = list(ds.shape)
                            ds_shape[-2:] = shape2d
                            ds_name_dict[key] = [ds.dtype, tuple(ds_shape), None]

        else:
            # construct dset structure from binary file
            ds_names = readfile.get_slice_list(ref_file)
            ds_dtype = meta['DATA_TYPE']
            for ds_name in ds_names:
                ds_name_dict[ds_name] = [ds_dtype, tuple(shape2d), None]

    # directory
    fdir = os.path.dirname(os.path.abspath(fname))
    if not os.path.isdir(fdir):
        os.makedirs(fdir)
        vprint('crerate directory: {}'.format(fdir))

    # create file
    with h5py.File(fname, "a") as f:
        vprint('create HDF5 file: {} with w mode'.format(fname))

        # initiate dataset
        max_digit = max([len(i) for i in ds_name_dict.keys()])
        for key in ds_name_dict.keys():
            data_type  = ds_name_dict[key][0]
            data_shape = ds_name_dict[key][1]

            # turn ON compression for conn comp
            ds_comp = compression
            if key in ['connectComponent']:
                ds_comp = 'lzf'

            # changable dataset shape
            if len(data_shape) == 3:
                max_shape = (None, data_shape[1], data_shape[2])
            else:
                max_shape = data_shape

            # create empty dataset
            vprint(("create dataset  : {d:<{w}} of {t:<25} in size of {s:<20} with "
                    "compression = {c}").format(d=key,
                                                w=max_digit,
                                                t=str(data_type),
                                                s=str(data_shape),
                                                c=ds_comp))
            if not key in f.keys():
                ds = f.create_dataset(key,
                                      shape=data_shape,
                                      maxshape=max_shape,
                                      dtype=data_type,
                                      chunks=True,
                                      compression=ds_comp)
            else:
                ds = f[key]

            # write auxliary data
            if len(ds_name_dict[key]) > 2 and ds_name_dict[key][2] is not None:
                ds[:] = np.array(ds_name_dict[key][2])

        # write attributes in root level
        for key, value in meta.items():
            f.attrs[key] = str(value)

        # write attributes in dataset level
        if ds_unit_dict is not None:
            for key, value in ds_unit_dict.items():
                if value is not None:
                    f[key].attrs['UNIT'] = value
                    vprint(f'add /{key:<{max_digit}} attribute: UNIT = {value}')

    vprint('close  HDF5 file: {}'.format(fname))

    return fname


def read_hdf5_file(fname, datasetName=None, box=None):
    """
    Parameters: fname : str, name of HDF5 file to read
                datasetName : str or list of str, dataset name in root level with/without date info
                    'timeseries'
                    'timeseries-20150215'
                    'unwrapPhase'
                    'unwrapPhase-20150215_20150227'
                    'HDFEOS/GRIDS/timeseries/observation/displacement'
                    'recons'
                    'recons-20150215'
                    ['recons-20150215', 'recons-20150227', ...]
                    '20150215'
                    'cmask'
                    'igram-20150215_20150227'
                    ...
                box : 4-tuple of int area to read, defined in (x0, y0, x1, y1) in pixel coordinate
    Returns:    data : 2D/3D array
                atr : dict, metadata
    """
    # File Info: list of slice / dataset / dataset2d / dataset3d
    slice_list = get_slice_list(fname)
    ds_list = []
    for i in [i.split('-')[0] for i in slice_list]:
        if i not in ds_list:
            ds_list.append(i)
    ds_2d_list = [i for i in slice_list if '-' not in i]
    ds_3d_list = [i for i in ds_list if i not in ds_2d_list]

    # Input Argument: convert input datasetName into list of slice
    if not datasetName:
        datasetName = [ds_list[0]]
    elif isinstance(datasetName, str):
        datasetName = [datasetName]
    if all(i.isdigit() for i in datasetName):
        datasetName = ['{}-{}'.format(ds_3d_list[0], i) for i in datasetName]
    # Input Argument: decompose slice list into dsFamily and inputDateList
    dsFamily = datasetName[0].split('-')[0]
    inputDateList = [i.replace(dsFamily,'').replace('-','') for i in datasetName]

    # read hdf5
    with h5py.File(fname, 'r') as f:
        # get dataset object
        dsNames = [i for i in [datasetName[0], dsFamily] if i in f.keys()]
        dsNamesOld = [i for i in slice_list if '/{}'.format(datasetName[0]) in i] # support for old mintpy files
        if len(dsNames) > 0:
            ds = f[dsNames[0]]
        elif len(dsNamesOld) > 0:
            ds = f[dsNamesOld[0]]
        else:
            raise ValueError('input dataset {} not found in file {}'.format(datasetName, fname))

        # 2D dataset
        if ds.ndim == 2:
            data = ds[box[1]:box[3], box[0]:box[2]]

        # 3D dataset
        elif ds.ndim == 3:
            # define flag matrix for index in time domain
            slice_flag = np.zeros((ds.shape[0]), dtype=np.bool_)
            if not inputDateList or inputDateList == ['']:
                slice_flag[:] = True
            else:
                date_list = [i.split('-')[1] for i in
                             [j for j in slice_list if j.startswith(dsFamily)]]
                for d in inputDateList:
                    slice_flag[date_list.index(d)] = True

            # read data
            data = ds[slice_flag, box[1]:box[3], box[0]:box[2]]
            data = np.squeeze(data)
    return data


def read(fname, box=None, datasetName=None, print_msg=True):
    """Read one dataset and its attributes from input file.
    Parameters: fname : str, path of file to read
                datasetName : str or list of str, slice names
                box : 4-tuple of int area to read, defined in (x0, y0, x1, y1) in pixel coordinate
    Returns:    data : 2/3-D matrix in numpy.array format, return None if failed
                atr : dictionary, attributes of data, return None if failed
    Examples:
        from mintpy.utils import readfile
        data, atr = readfile.read('velocity.h5')
        data, atr = readfile.read('timeseries.h5')
        data, atr = readfile.read('timeseries.h5', datasetName='timeseries-20161020')
        data, atr = readfile.read('ifgramStack.h5', datasetName='unwrapPhase')
        data, atr = readfile.read('ifgramStack.h5', datasetName='unwrapPhase-20161020_20161026')
        data, atr = readfile.read('ifgramStack.h5', datasetName='coherence', box=(100,1100, 500, 2500))
        data, atr = readfile.read('geometryRadar.h5', datasetName='height')
        data, atr = readfile.read('geometryRadar.h5', datasetName='bperp')
        data, atr = readfile.read('100120-110214.unw', box=(100,1100, 500, 2500))
    """
    # metadata
    dsname4atr = None   #used to determine UNIT
    if isinstance(datasetName, list):
        dsname4atr = datasetName[0].split('-')[0]
    elif isinstance(datasetName, str):
        dsname4atr = datasetName.split('-')[0]

    files = [fname + i for i in ['.rsc', '.xml']]
    fext0 = ['.' + i.split('.')[-1] for i in files if os.path.exists(i)][0]

    atr = read_attribute(fname, datasetName=dsname4atr, metafile_ext=fext0)

    # box
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    if not box:
        box = (0, 0, width, length)

    # Read Data
    fext = os.path.splitext(os.path.basename(fname))[1].lower()
    if fext in ['.h5', '.he5']:
        data = read_hdf5_file(fname, datasetName=datasetName, box=box)
    else:
        data, atr = read_binary_file(fname, datasetName=datasetName, box=box)
        #data, atr = readfile.read_binary_file(fname, datasetName=datasetName, box=box, xstep=1, ystep=1)
    return data, atr
