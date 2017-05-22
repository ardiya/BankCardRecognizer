from flask import Flask, request, jsonify, send_from_directory
from scipy import misc
import numpy
import operator
import os
import re
import shutil

app = Flask(__name__)
app.debug = True

import neuralnet

#Web interface
@app.route("/")
def hello():
    return """<h2>Machine Learning-Image Recognition</h2>
<form enctype="multipart/form-data" action="/upload" method="POST">
    Bank Card Image:<input type="file" name="image"><br/>
    <button>Upload</button>
</form>"""

#Image to be show on andoid APP
@app.route('/img/<path:path>')
def send_js(path):
    return send_from_directory('E:\\imgsave\\', path)

#Web Service
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if None == request.files['image']: return jsonify({'success': False})
        #Save Image
        request.files['image'].save('IMG_0193.JPG')
		
        img_path = "E:\\imgsave\\"
        try:
            shutil.rmtree(img_path)
        except Exception, e:
            pass
        finally:
            os.mkdir(img_path)
		
        #run opencv code
        os.system("BankCard_Final.exe")
		
        regex = r"wnd_(?P<id>\d+).jpg"
        list_train_img = os.listdir(img_path)
        result = list()
        for img in list_train_img:
            #get the id of digit
            img_id = int(re.match(regex, img).group('id'))
            
            #Convert image into 2D-matrix
            jpg = misc.imread(img_path+img, flatten=True)
            jpg = misc.imresize(jpg, (28,28))
            
            #Forward Propagation data from image
            l0 = jpg.flatten()
            l1 = neuralnet.sigmoid(numpy.dot(l0, neuralnet.weight0))
            l2 = neuralnet.sigmoid(numpy.dot(l1, neuralnet.weight1))
            idx, val = max(enumerate(l2), key=operator.itemgetter(1))
            result.append({'img_name':img,'img_id':img_id,'prediction':idx,'confidence':val})
        
        result.sort(key=lambda x: x['img_id'])
        
        #Getting the full bank card number
        bankNumber = ""
        for r in result:
            bankNumber+=str(r['prediction'])
        
        return jsonify({'success': True,'bank_number':bankNumber,'payload':result})


if __name__ == "__main__":
    app.run(host='0.0.0.0')
    
