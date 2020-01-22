import os
import multiprocessing as mp
import sys

if __name__ == '__main__':
    sys.path.append(os.getcwd())

    import importing.files as files


def main():

    # An instance of files
    images = files.Files()

    # Delete any existing image files
    images.cleanup()

    # Ensure that the directory that would host the unzipped images exists
    images.local_directory()

    # Get the iterable list of zipped files names
    blobs = files.Files().list_of_blobs()

    # Download and unzip; to save time, this is done in parallel
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(files.Files().extractor, (i for i in blobs))


if __name__ == '__main__':
    main()
