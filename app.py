import tensorflow as tf
import flask
import werkzeug
import my_model as my_mod
from objectdet.TensorFlow.models.research.object_detection import obj
from objectdet.TensorFlow.models.research.object_detection.obj import obj_det

#import cv2
#import matplotlib.pyplot as plt

app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    hg=my_mod.model_indoor()
    sd=obj.myvar
    desr=obj_det()
    dfg=sd+" "+hg+" "+desr
    print(dfg)



    return dfg


app.run(host="0.0.0.0", port=5000, debug=True)
