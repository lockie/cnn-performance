#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import random
import sys
import os
import os.path
from struct import unpack
from array import array
from urlparse import urlparse
import shutil
import numpy as np
from scipy.misc import imread

IMAGES_SIZE = 108
BATCH_SIZE = 1000


def get_file(url):
    filename = os.path.basename(urlparse(url).path)
    if not os.path.isfile(filename):
        import urllib
        urllib.URLopener().retrieve(url, filename)

def get_files():
    print "Downloading & unpacking dataset..."

    get_file('http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/'
            'norb-5x46789x9x18x6x2x108x108-training-01-cat.mat.gz')
    get_file('http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/'
            'norb-5x46789x9x18x6x2x108x108-training-01-dat.mat.gz')
    get_file('http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/'
            'norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat.gz')
    get_file('http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/'
            'norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat.gz')

    # TODO : use GzipFile of some sort
    if not os.path.isfile('norb-5x46789x9x18x6x2x108x108-training-01-cat.mat'):
        import subprocess
        subprocess.call(['gzip', '-fkd',
            'norb-5x46789x9x18x6x2x108x108-training-01-cat.mat.gz'])
    if not os.path.isfile('norb-5x46789x9x18x6x2x108x108-training-01-dat.mat'):
        import subprocess
        subprocess.call(['gzip', '-fkd',
            'norb-5x46789x9x18x6x2x108x108-training-01-dat.mat.gz'])
    if not os.path.isfile('norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat'):
        import subprocess
        subprocess.call(['gzip', '-fkd',
            'norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat.gz'])
    if not os.path.isfile('norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat'):
        import subprocess
        subprocess.call(['gzip', '-fkd',
            'norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat.gz'])

    print "done."


data_mean = None
data_count = 0

def create_batch_file(data_list, filename):
    global data_mean
    print "Writing batch file '" + filename + "' ..."
    N = len(data_list)
    data = np.zeros((IMAGES_SIZE*IMAGES_SIZE, N), dtype=np.uint8)
    labels = [0] * N
    for i in xrange(N):
        data[:,i] = data_list[i][0]
        data_mean += data_list[i][0]
        labels[i] = data_list[i][1]

    import cPickle
    f = open(filename, 'wb')
    cPickle.dump({'data': data, 'labels': labels}, f)
    f.close()


def process_files(cat_file_name, dat_file_name, batchfile_basename, datadir):
    global data_count

    catfile = open(cat_file_name, 'rb')
    datfile = open(dat_file_name, 'rb')

    catheader = catfile.read(20)
    datheader = datfile.read(20)

    magic, ndim, dim0, dim1, dim2 = unpack('iiiii', catheader)
    if magic != 0x1e3d4c54: # integer matrix
        print "Unsupported data type on catfile!"
        sys.exit(1)
    if ndim != 1:
        print "Incorrect cat file dimension count:", ndim
        sys.exit(1)
    N = dim0
    magic, ndim, dim0, dim1, dim2 = unpack('iiiii', datheader)
    if magic != 0x1e3d4c55: # byte matrix
        print "Unsupported data type on datfile!"
        sys.exit(1)
    if ndim != 4:
        print "Incorrect dat file dimension count:", ndim
        sys.exit(1)
    if dim0 != N:
        print "cat file does not match dat file"
        sys.exit(1)
    dim3 = unpack('i', datfile.read(4))[0]
    if dim2 != IMAGES_SIZE or dim3 != IMAGES_SIZE:
        print "Incorrect image size: ", dim2, "x", dim3
        sys.exit(1)

    print "Processing", N, "images..."

    for i in xrange(N/BATCH_SIZE):
        data = []
        for j in xrange(BATCH_SIZE):
            label = unpack('i', catfile.read(4))[0]
            image = np.array(array('B', datfile.read(
                    IMAGES_SIZE*IMAGES_SIZE))).astype(np.uint8, copy=False)
            data.append((image, label))
        data_count += BATCH_SIZE
        random.shuffle(data)
        create_batch_file(data, os.path.join(datadir,
                "{0}_{1}".format(batchfile_basename, i)))
    catfile.close()
    datfile.close()


def write_metafile(datadir):
    import cPickle
    f = open(os.path.join(datadir, "batches.meta"), 'wb')
    cPickle.dump({
        'label_names': ['animal', 'human', 'airplane', 'truck', 'car', 'non-object'],
        'data_mean': (data_mean*(1./data_count)).astype('uint8').reshape((IMAGES_SIZE*IMAGES_SIZE, 1))
    },
    f)
    f.close()

def cleanup_files():
    for fn in ['norb-5x46789x9x18x6x2x108x108-training-01-dat.mat',
            'norb-5x46789x9x18x6x2x108x108-training-01-cat.mat',
            'norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat',
            'norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat']:
        if os.path.isfile(fn):
            os.remove(fn)

def main():
    datadir = "."
    if len(sys.argv) > 1:
        datadir = sys.argv[1]
    do_stuff = False
    if not os.path.exists(datadir):
        do_stuff = True
        os.makedirs(datadir)
    global data_mean
    data_mean = np.zeros(IMAGES_SIZE*IMAGES_SIZE, dtype=np.float)
    try:
        if do_stuff:
            get_files()
            process_files('norb-5x46789x9x18x6x2x108x108-training-01-cat.mat',
                    'norb-5x46789x9x18x6x2x108x108-training-01-dat.mat',
                    'data_batch', datadir)
            process_files('norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat',
                    'norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat',
                    'test_batch', datadir)
            write_metafile(datadir)
    except BaseException as e:
        shutil.rmtree(datadir)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup_files()



if __name__ == '__main__':
    main()

