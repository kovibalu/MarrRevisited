import os
import numpy as np
from PIL import Image


def progress_bar(l, show_progress=True):
    """ Returns an iterator for a list or queryset that renders a progress bar
    with a countdown timer """
    if show_progress:
        return iterator_progress_bar(l)
    else:
        return l


def iterator_progress_bar(iterator, maxval=None):
    """ Returns an iterator for an iterator that renders a progress bar with a
    countdown timer """

    from progressbar import ProgressBar, SimpleProgress, Bar, ETA
    pbar = ProgressBar(
        maxval=maxval,
        widgets=[SimpleProgress(sep='/'), ' ', Bar(), ' ', ETA()],
    )
    return pbar(iterator)


def main():
    image_count = 1449
    normals = np.empty((image_count, 480, 640, 3))
    train_idxs = []
    test_idxs = []

    for i in progress_bar(xrange(1449)):
        nyu_idx = i + 1
        fname = '%05d.png' % nyu_idx
        fpath = os.path.join('train', fname)
        if os.path.exists(fpath):
            train_idxs.append(nyu_idx)
        else:
            fpath = os.path.join('test', fname)
            assert os.path.exists(fpath)
            test_idxs.append(nyu_idx)

        normals[i] = np.array(Image.open(fpath)) / 255.

    np.save('normals', normals)
    np.save('train_idxs', train_idxs)
    np.save('test_idxs', test_idxs)


if __name__ == '__main__':
    main()
