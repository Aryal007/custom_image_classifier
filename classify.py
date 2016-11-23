"""Classifies image arguments 

"""

import json
import sys
import os

from skimage import io
from skimage import transform

from model import setup_model

SIZE = (32, 32)
MODEL_FILE = "classifier.tfl"

model = setup_model()

model.load(MODEL_FILE)

filenames = sys.argv[1:]

for filename in filenames:
    filepath = os.path.abspath(filename)
    try:
        im = io.imread(filepath)
        im = transform.resize(im, SIZE)
    except ValueError:
        print("Error")
        # print("Unable to load: {:s}".format(filepath), file = sys.stderr)
    result = model.predict([im])
    print(json.dumps({
        "filepath": filepath,
        "Triangle_score": result[0][0],
        "Circle_score": result[0][1]
    }))
