import io
import os
import sys
import zipfile

import numpy as np
import requests

import config


class Files:

    def __init__(self):

        self.name = 'Files'

        variables = config.Config().variables()

        self.images_path = variables['images']['path']

        zipped_images = variables['zipped']['images']
        self.url = zipped_images['url']
        self.starting = zipped_images['from']
        self.ending = zipped_images['to']
        self.ext = zipped_images['ext']
        self.zero_filling = zipped_images['zero_filling']


    def cleanup(self):

        [os.remove(os.path.join(base, file))
         for base, directories, files in os.walk(self.images_path)
         for file in files]

        [os.removedirs(os.path.join(base, directory))
         for base, directories, files in os.walk(self.images_path, topdown=False)
         for directory in directories
         if os.path.exists(os.path.join(base, directory))]


    def local_directory(self):

        if os.path.exists(self.images_path):
            os.mkdir(self.images_path)


    def extractor(self, blob):
        try:
            req = requests.get(self.url + blob + self.ext)
        except OSError as err:
            print(err)
            sys.exit(1)

        zipped_object = zipfile.ZipFile(io.BytesIO(req.content))
        zipped_object.extractall(path='images')


    def list_of_blobs(self):

        # This is peculiar ...
        blobs = [str(i).zfill(self.zero_filling) for i in np.arange(self.starting, self.ending)]

        # Very peculiar ...
        return [{blob} for blob in blobs]
