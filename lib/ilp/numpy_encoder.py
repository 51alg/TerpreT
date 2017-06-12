#from http://stackoverflow.com/questions/3488934/simplejson-and-numpy-array/24375113#24375113
import base64
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)