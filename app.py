from flask import Flask, request, redirect, url_for, render_template
from flask_uploads import UploadSet, configure_uploads, ALL
import os
import json
import glob
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import logging
import os
import re
import sys
import time
import pprint
from datetime import datetime
import mimetypes
from flask import Response, render_template
from flask import Flask
from flask import send_file
from flask import request
from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os,sys
import traceback
import subprocess
import socket
import numpy as np
from dataset.preprocess_data import *
from PIL import Image, ImageFilter,ImageDraw
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import time
import torch.utils
import sys
from utils import *
import pdb
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
import time


frames=[]
cls_dict={
    0:'Abuse',
    1:'Burglary',
2:'Explosion',
3:'Fight',
4:'Normal',
5:'RoadAccidents',
6:'Robbery',
7:'Shooting',
8:'Shoplifting',
9:'Vandalism',
10:'Test'
}
app = Flask(__name__)
LOG = logging.getLogger(__name__)
VIDEO_PATH = '/video'

MB = 1 << 20
BUFF_SIZE = 10 * MB


files = UploadSet('files', ALL)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD']=True
app.config['UPLOADED_FILES_DEST'] = 'uploadr/static'
configure_uploads(app, files)

opt = parse_opts()
opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
if opt.modality=='RGB': opt.input_channels = 3
elif opt.modality=='Flow': opt.input_channels = 2
inp='static/tmp.mp4'
lb='0'
clip1=extract_images(inp)
model, parameters = generate_model(opt)
if opt.resume_path1:
    print('loading checkpoint {}'.format(opt.resume_path1))
    checkpoint = torch.load(opt.resume_path1,map_location={'cuda:0': 'cpu'})
    assert opt.arch == checkpoint['arch']
    model.load_state_dict(checkpoint['state_dict'])
model.eval()	

def make_video(images, outimg=None, fps=16, size=None,is_color=True, format="XVID"):
    fourcc=VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if vid is None:
            if size is None:
                size = image.shape[1], image.shape[0]
                print("SIZEEEEE",size)
            vid = VideoWriter("ok.avi", fourcc, float(fps), size, is_color)
        if size[0] != image.shape[1] and size[1] != image.shape[0]:
            image = resize(image, size)
        vid.write(image)
    vid.release()
    print("SIZEEEEE*******************",size)
    return vid

def extract_images(video_input_file_path,lb=0,outdir='/home/ali4426623/Test'):
    os.system('rm -rf %s/*' % outdir)
    try:
        print("OK0")
            # check if horizontal or vertical scaling factor
        o = subprocess.check_output('ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"'%(video_input_file_path), shell=True).decode('utf-8')
        print("OK1")
        lines = o.splitlines()
        width = int(lines[0].split('=')[1])
        height = int(lines[1].split('=')[1])
        resize_str = '-1:256' if width>height else '256:-1'
        print("OK2")
        os.system('ffmpeg -i "%s" -r 25 -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1'%( video_input_file_path, resize_str,os.path.join(outdir, '%05d.jpg') ))
        nframes = len([ fname for fname in os.listdir(outdir) if fname.endswith('.jpg') and len(fname)==9])
        if nframes==0: raise Exception 
        print("OK3")
        os.system('touch "%s"'%(os.path.join(outdir, 'done') ))
    except Exception:
        traceback.print_exc()
    frame_path='/home/jupyter/HammadData/Frames/Test/Test'
    total=len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))
    clip1=get_test_video(opt,frame_path,total) 
    return((scale_crop(clip1, 0, opt), lb))


@app.route("/")
def index():
    return render_template("index.html")
@app.route("/signin")
def signin():
    return render_template("sign-in.html")
def createaccount():
    return render_template("createaccount.html")
@app.route("/add-video",methods=["GET","POST"])
def addvideo():
    if request.method == 'POST' and 'media' in request.files:
        if os.path.exists('uploadr/static/tmp.mp4'):
            os.remove("uploadr/static/tmp.mp4")
        filename = files.save(request.files['media'],name='tmp.mp4')
        with torch.no_grad():
            clip = torch.squeeze(clip1[0])
            if opt.modality == 'RGB':
                inputs = torch.Tensor(int(clip.shape[1]/opt.sample_duration), 3, opt.sample_duration, opt.sample_size, opt.sample_size)
            elif opt.modality == 'Flow':
                inputs = torch.Tensor(int(clip.shape[1]/opt.sample_duration), 2, opt.sample_duration, opt.sample_size, opt.sample_size)
            for k in range(inputs.shape[0]):
                inputs[k,:,:,:,:] = clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]
            inputs_var = Variable(inputs)
            outputs_var= model(inputs_var)
            pred5 = np.array(torch.mean(outputs_var, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            files=sorted(glob.glob('/home/ali4426623/Test/*.jpg'))
            op = [cv2.imread(i) for i in files]
            for idx,ii in enumerate(outputs_var):
                ind = (-ii).argsort()[:4]
                current=op[idx*opt.sample_duration:(idx+1)*opt.sample_duration]
                p=0
                for pp in ind:
                    p+=ii[pp]
                total=p
                for i in current:
                    li=ind.tolist()
                    cc=[cls_dict[i] for i in li]
                    cv2.putText(i, str(cc[0])+'='+str(int(ii[ind[0]]*100/total)), (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 1)
                    cv2.putText(i, str(cc[1])+'='+str(int(ii[ind[1]]*100/total)), (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 1)
                    cv2.putText(i, str(cc[2])+'='+str(int(ii[ind[2]]*100/total)), (20, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 1)
                    cv2.putText(i, str(cc[3])+'='+str(int(ii[ind[3]]*100/total)), (20, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 1)
                    frames.append(i)
                print('Predicted:',ind.tolist())
            print(make_video(frames))
    return render_template('add-video.html')
@app.route("/camera-survillance")
def camerasurvillance():
    return render_template("camera-survillance.html")

@app.route('/show_video')
def home():
    LOG.info('Rendering home page')
    response = render_template(
        'show_video.html',
		time=str(datetime.now()),
        video=VIDEO_PATH,
    )
    return response

def partial_response(path, start, end=None):
    LOG.info('Requested: %s, %s', start, end)
    file_size = os.path.getsize(path)

    # Determine (end, length)
    if end is None:
        end = start + BUFF_SIZE - 1
    end = min(end, file_size - 1)
    end = min(end, start + BUFF_SIZE - 1)
    length = end - start + 1

    # Read file
    with open(path, 'rb') as fd:
        fd.seek(start)
        bytes = fd.read(length)
    assert len(bytes) == length

    response = Response(
        bytes,
        206,
        mimetype=mimetypes.guess_type(path)[0],
        direct_passthrough=True,
    )
    response.headers.add(
        'Content-Range', 'bytes {0}-{1}/{2}'.format(
            start, end, file_size,
        ),
    )
    response.headers.add(
        'Accept-Ranges', 'bytes'
    )
    LOG.info('Response: %s', response)
    LOG.info('Response: %s', response.headers)
    return response

def get_range(request):
    range = request.headers.get('Range')
    LOG.info('Requested: %s', range)
    m = re.match('bytes=(?P<start>\d+)-(?P<end>\d+)?', range)
    if m:
        start = m.group('start')
        end = m.group('end')
        start = int(start)
        if end is not None:
            end = int(end)
        return start, end
    else:
        return 0, None

@app.route(VIDEO_PATH)
def video():
    path = 'uploadr/static/tmp.mp4'
#    path = 'demo.mp4'

    start, end = get_range(request)
    return partial_response(path, start, end)
if __name__ == '__main__':
    flask_options = dict(
        host='0.0.0.0',
        debug=True,
        port=2007,
        threaded=True,
    )
    logging.basicConfig(level=logging.INFO)
    app.run(**flask_options)
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(8080)
    IOLoop.instance().start()

