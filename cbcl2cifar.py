#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import random
import sys
import os
import os.path
import shutil
import numpy as np
from scipy.misc import imread

IMAGES_SIZE = 19
BATCH_SIZE = 1000


def get_files():
    if not os.path.isfile('faces.tar.gz'):
        print "Downloading dataset..."
        import urllib
        urllib.URLopener().retrieve(
            'http://cbcl.mit.edu/projects/cbcl/software-datasets/faces.tar.gz',
            'faces.tar.gz')
        print "done."
    if not os.path.isfile('face.test.tar.gz') or not os.path.isfile('face.train.tar.gz'):
        import subprocess
        subprocess.call(['tar', 'xf', 'faces.tar.gz'])
    if not os.path.isdir('test') or not os.path.isdir('train'):
        import subprocess
        subprocess.call(['tar', 'xf', 'face.test.tar.gz'])
        subprocess.call(['tar', 'xf', 'face.train.tar.gz'])

data_mean = None
data_count = 0

def create_batch_file(data_list, filename):
    global data_mean
    print "Writing batch file '" + filename + "' ..."
    N = len(data_list)
    data = np.zeros((IMAGES_SIZE*IMAGES_SIZE, N), dtype=np.uint8)
    labels = [0] * N
    for i in xrange(N):
        img = imread(data_list[i][0]).reshape(IMAGES_SIZE*IMAGES_SIZE)
        data[:,i] = img
        data_mean += np.array(img)
        labels[i] = data_list[i][1]

    import cPickle
    f = open(filename, 'wb')
    cPickle.dump({'data': data, 'labels': labels}, f)
    f.close()


def process_directory(directory, batchfile_basename, datadir):
    global data_count
    data = []
    path = os.path.join(os.getcwd(), directory, "face")
    for f in os.listdir(path):
        data.append((os.path.join(path, f), 1))
    path = os.path.join(os.getcwd(), directory, "non-face")
    for f in os.listdir(path):
        data.append((os.path.join(path, f), 0))
    random.shuffle(data)
    data_count += len(data)
    index = 0
    for chunk in [data[i:i+BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]:
        create_batch_file(chunk, os.path.join(datadir,
            "{0}_{1}".format(batchfile_basename, index)))
        index += 1

def write_metafile(datadir):
    import cPickle
    f = open(os.path.join(datadir, "batches.meta"), 'wb')
    cPickle.dump({
        'label_names': ['non-face', 'face'],
        'data_mean': (data_mean*(1./data_count)).astype('uint8').reshape((IMAGES_SIZE*IMAGES_SIZE, 1))
    },
    f)
    f.close()

def cleanup_files():
    for fn in [
            'README', 'svm.test.normgrey', 'svm.train.normgrey',
            'face.test.tar.gz', 'face.train.tar.gz']:
        if os.path.isfile(fn):
            os.remove(fn)
    shutil.rmtree('test')
    shutil.rmtree('train')

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
        get_files()
        if do_stuff:
            process_directory("train", "data_batch", datadir)
            process_directory("test", "test_batch", datadir)
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

