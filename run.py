from flask_uploads import UploadSet, configure_uploads, ALL
import os
from itertools import groupby
from operator import itemgetter
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
from camera import VideoCamera
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
from flask_login import UserMixin
from itertools import groupby
import sqlite3
from flask import Flask, render_template, request, flash, redirect, url_for,session

app = Flask(__name__)
LOG = logging.getLogger(__name__)
MB = 1 << 20
BUFF_SIZE = 10 * MB


files = UploadSet('files', ALL)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
#app.config['TEMPLATES_AUTO_RELOAD']=True
app.config['UPLOADED_FILES_DEST'] = '/home/jupyter/all_v3/all_v1/final_backup/Done/static/all_videos'
configure_uploads(app, files)
app.config['SECRET_KEY'] = 'this should be a secret random string'



class GroupedFrames:
    def __init__(self,current_frames,respective_score_list,classes):
        self.current_frames = current_frames
        self.current_list=respective_score_list
        self.current_list_classes=classes

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


opt = parse_opts()
opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
if opt.modality=='RGB': opt.input_channels = 3
elif opt.modality=='Flow': opt.input_channels = 2
inp=None
lb='0'
model, parameters = generate_model(opt)
if opt.resume_path1:
    print('loading checkpoint {}'.format(opt.resume_path1))
    checkpoint = torch.load(opt.resume_path1,map_location={'cuda:0': 'cpu'})
    assert opt.arch == checkpoint['arch']
    model.load_state_dict(checkpoint['state_dict'])
model.eval()	

def make_video(images,name,outname, outimg=None, fps=16, size=None,is_color=True, format="XVID"):
    fourcc=VideoWriter_fourcc(*format)
    vid = None
    print("IMAGES SIZZE",len(images))
    for image in images:
        if vid is None:
            if size is None:
                size = image.shape[1], image.shape[0]
                print("SIZEEEEE",size)
            print("NEW!!")
            vid = VideoWriter(name, fourcc, float(fps), size, is_color)
        if size[0] != image.shape[1] and size[1] != image.shape[0]:
            image = resize(image, size)
        vid.write(image)
    vid.release()
    os.system("ffmpeg -i "+ name+" -strict -2 "+outname)
    os.system("rm "+name)
    return vid

def extract_images(video_input_file_path,lb=0,outdir='/home/jupyter/HammadData/Frames/Test/Test'):
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

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def fetch_data(command):
    conn = get_db_connection()
    todos = conn.execute(command).fetchall()
    return todos

@app.route("/")
def index():
    cmd='select id,videos,created,action from videos where user_id="{0}"'.format(str(session['id']))
    aaa=fetch_data(cmd)
    rows=[]
    for i in aaa:
        rows.append([i['id'],i['videos'].split("|")[0].rpartition('/')[2],i['created'],i['action']])
    return render_template("index.html",rows=rows)
def add_video_db(cmd):
        connection = sqlite3.connect('database.db')
        with open('schema.sql') as f:
            connection.executescript(f.read())
        cur = connection.cursor()
        cur.execute(cmd)
        connection.commit()
        connection.close()
        
@app.route("/signin",methods=["GET","POST"])
def signin():
    if request.method == 'POST':
        username,password=request.form['username'],request.form['password']
        print((username,password))
        command=st='select id,username,password from users where username="{0}" and password="{1}"'.format(username,password)
        vals=fetch_data(command)
        print(vals)
        if len(vals)==0:
            return render_template("sign-in.html",error="USERNAME/PASSWORD NOT FOUND!")
        else:
            session['loggedin'] = True
            session['id'] = vals[0]['id']
            session['username'] = vals[0]['username']
            return redirect(url_for('index'))
    return render_template("sign-in.html")
@app.route("/signup",methods=["GET","POST"])
def signup():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        ff=fetch_data('select * from users where username="{0}"'.format(username))
        print(ff)
        if len(ff)>=1:
            return render_template("createaccount.html",username_error='Email already registerd!')
        else:
            cmd='insert into users(username,password) values("{0}","{1}")'.format(username,password)
            add_video_db(cmd)
            return redirect(url_for('signin'))
            
    return render_template("createaccount.html")
@app.route("/add-video",methods=["GET","POST"])
def addvideo():
    base_name='/home/jupyter/all_v3/all_v1/final_backup/Done/'
    cam_id = request.args.get('cam_id')
    if session['loggedin']:
        hash = random.getrandbits(128)
        db_in=[]
        count=0
        if request.method == 'POST' and 'media' in request.files:
            start=time.time()
            os.system('rm -rf /home/jupyter/HammadData/Test/Test/*')
            os.system("rm -rf /home/jupyter/HamadData/Frames/Test/Test/*")
			os.system("rm -rf /home/jupyter/all_v3/all_v1/final_backup/Done/static/tmp.mp4")
            name='static/saved/{0}.mp4'.format(str(hash))
            db_in.append(name)
            name=base_name+name
            filename =files.save(request.files['media'],name='/home/jupyter/all_v3/all_v1/final_backup/Done/static/tmp.mp4')
            frames=[]
            all11=[]
            all12=[]
            last=[]
            clip1=extract_images('/home/jupyter/all_v3/all_v1/final_backup/Done/static/tmp.mp4')
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

                files1=sorted(glob.glob('/home/jupyter/HammadData/Frames/Test/Test/*.jpg'))
                op = [cv2.imread(i) for i in files1]
                for idx,ii in enumerate(outputs_var):
                    ind = (-ii).argsort()[:4]
                    all11.append([ii[ind[0]],ii[ind[1]],ii[ind[2]],ii[ind[3]]])
                    all12.append(int(ind[0]))
                    current=op[idx*opt.sample_duration:(idx+1)*opt.sample_duration]
                    last.append((int(ind[0]),GroupedFrames(current,[ii[ind[0]],ii[ind[1]],ii[ind[2]],ii[ind[3]]],[ind[0],ind[1],ind[2],ind[3]])))
                    p=0
                    for pp in ind:
                        p+=ii[pp]
                    total=p
                    for i in current:
                        li=ind.tolist()
                        cc=[cls_dict[i] for i in li]
                        cv2.putText(i, str(cc[0])+'='+str(int(ii[ind[0]]*100/total)), (20, 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255), 1)

                        cv2.putText(i, str(cc[1])+'='+str(int(ii[ind[1]]*100/total)), (20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255), 1)
                        cv2.putText(i, str(cc[2])+'='+str(int(ii[ind[2]]*100/total)), (20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255), 1)
                        cv2.putText(i, str(cc[3])+'='+str(int(ii[ind[3]]*100/total)), (20, 80),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255), 1)
                        frames.append(i)
                        count+=1
                img_name='static/saved/{0}.png'.format(hash)
                make_video(frames,name='/home/jupyter/all_v3/all_v1/final_backup/Done/static/tmp.mp4',outname=name)
                cv2.imwrite(img_name,frames[0])
                db_in.append('saved/{0}.png'.format(hash))
                f = open("static/all_videos/ok.txt", "w")
                prob=str(str(cc[0])+" "+str(cc[1])+" "+str(cc[2])+" "+str(cc[3]))
                prob1=str(str(cc[0])+","+str(cc[1])+","+str(cc[2])+","+str(cc[3]))
                f.write(prob)
                f.close()
                db_in.append(prob1)
                res = [list(map(itemgetter(1), temp)) for (key, temp) in groupby(last, itemgetter(0))]
                res_next = [list(map(itemgetter(0), temp)) for (key, temp) in groupby(last, itemgetter(0))]
                total=0
                ccc=0
                res1=[]
                for indx,i in enumerate(res_next):
                    if len(i) > 4 and i[0]!=4:
                        res1.append(res[indx])
                print(len(res1),res1)
                count=0
                for element in res1:
                    temp=[]
                    cls_lbl=None
                    for indx,j in enumerate(element):
                        current=j.current_frames
                        scores=j.current_list
                        list_class=j.current_list_classes
                        total=0
                        for i in scores:
                            total=total+i
                        for i in current:
                            cc=[cls_dict[int(i)] for i in list_class]
                            temp.append(i)
                            cls_lbl=cc[0]
                    name=base_name+str('static/saved/'+(str(hash)+str(count))+'.avi')
                    outname=base_name+str('static/saved/'+(str(hash)+str(count))+'.mp4')
                    make_video(temp,name=name,outname=outname)
                    cv2.imwrite('static/saved/'+(str(hash)+str(count))+'.png',temp[0])
                    db_in.append('static/saved/'+(str(hash)+str(count))+'.mp4')
                    db_in.append('saved/'+(str(hash)+str(count))+'.png')
                    count+=1
        if len(db_in)>0:
            st=""
            for i in db_in:
                st=st+str(i)+"|"
            st=st[:-1]
            action=str(db_in[2].split(',')[0])
            command='INSERT INTO videos(user_id,videos,action) VALUES("{0}","{1}","{2}")'.format(str(session['id']),st,action)
            print(command)
            add_video_db(command)

        return render_template('add-video.html')
    else:
        return redirect(url_for('signin'))
@app.route("/camera-survillance",methods=["GET","POST"])
def camerasurvillance():
    base_name='/home/jupyter/all_v3/all_v1/final_backup/Done/'
    if session['loggedin']:
        hash = random.getrandbits(128)
        db_in=[]
        count=0
        if request.method == 'POST' and 'media' in request.files:
            start=time.time()
            os.system('rm -rf /home/jupyter/HammadData/Test/Test/*')
            os.system("rm -rf /home/jupyter/HamadData/Frames/Test/Test/*")
			os.system("rm -rf /home/jupyter/all_v3/all_v1/final_backup/Done/static/tmp.mp4")
            name='static/saved/{0}.mp4'.format(str(hash))
            db_in.append(name)
            name=base_name+name
            filename =files.save(request.files['media'],name='/home/jupyter/all_v3/all_v1/final_backup/Done/static/tmp.mp4')
            frames=[]
            all11=[]
            all12=[]
            last=[]
            clip1=extract_images('/home/jupyter/all_v3/all_v1/final_backup/Done/static/tmp.mp4')
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

                files1=sorted(glob.glob('/home/jupyter/HammadData/Frames/Test/Test/*.jpg'))
                op = [cv2.imread(i) for i in files1]
                for idx,ii in enumerate(outputs_var):
                    ind = (-ii).argsort()[:4]
                    all11.append([ii[ind[0]],ii[ind[1]],ii[ind[2]],ii[ind[3]]])
                    all12.append(int(ind[0]))
                    current=op[idx*opt.sample_duration:(idx+1)*opt.sample_duration]
                    last.append((int(ind[0]),GroupedFrames(current,[ii[ind[0]],ii[ind[1]],ii[ind[2]],ii[ind[3]]],[ind[0],ind[1],ind[2],ind[3]])))
                    p=0
                    for pp in ind:
                        p+=ii[pp]
                    total=p
                    for i in current:
                        li=ind.tolist()
                        cc=[cls_dict[i] for i in li]
                        cv2.putText(i, str(cc[0])+'='+str(int(ii[ind[0]]*100/total)), (20, 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255), 1)

                        cv2.putText(i, str(cc[1])+'='+str(int(ii[ind[1]]*100/total)), (20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255), 1)
                        cv2.putText(i, str(cc[2])+'='+str(int(ii[ind[2]]*100/total)), (20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255), 1)
                        cv2.putText(i, str(cc[3])+'='+str(int(ii[ind[3]]*100/total)), (20, 80),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255), 1)
                        frames.append(i)
                        count+=1
                img_name='static/saved/{0}.png'.format(hash)
                tmp_name='/home/jupyter/all_v3/all_v1/final_backup/Done/static/tmp.mp4'
                make_video(frames,name=tmp_name,outname=name)
                cv2.imwrite(img_name,frames[0])
                db_in.append('saved/{0}.png'.format(hash))
                f = open("static/all_videos/ok.txt", "w")
                prob=str(str(cc[0])+" "+str(cc[1])+" "+str(cc[2])+" "+str(cc[3]))
                prob1=str(str(cc[0])+","+str(cc[1])+","+str(cc[2])+","+str(cc[3]))
                f.write(prob)
                f.close()
                db_in.append(prob1)
                res = [list(map(itemgetter(1), temp)) for (key, temp) in groupby(last, itemgetter(0))]
                res_next = [list(map(itemgetter(0), temp)) for (key, temp) in groupby(last, itemgetter(0))]
                total=0
                ccc=0
                res1=[]
                for indx,i in enumerate(res_next):
                    if len(i) > 4 and i[0]!=4:
                        res1.append(res[indx])
                print(len(res1),res1)
                count=0
                for element in res1:
                    temp=[]
                    cls_lbl=None
                    for indx,j in enumerate(element):
                        current=j.current_frames
                        scores=j.current_list
                        list_class=j.current_list_classes
                        total=0
                        for i in scores:
                            total=total+i
                        for i in current:
                            cc=[cls_dict[int(i)] for i in list_class]
                            temp.append(i)
                            cls_lbl=cc[0]
                    name=base_name+str('static/saved/'+(str(hash)+str(count))+'.avi')
                    outname=base_name+str('static/saved/'+(str(hash)+str(count))+'.mp4')
                    make_video(temp,name=name,outname=outname)
                    cv2.imwrite('static/saved/'+(str(hash)+str(count))+'.png',temp[0])
                    db_in.append('static/saved/'+(str(hash)+str(count))+'.mp4')
                    db_in.append('saved/'+(str(hash)+str(count))+'.png')
                    count+=1
        if len(db_in)>0:
            st=""
            for i in db_in:
                st=st+str(i)+"|"
            st=st[:-1]
            action=str(db_in[2].split(',')[0])
            command='INSERT INTO videos(user_id,videos,action) VALUES("{0}","{1}","{2}")'.format(str(session['id']),st,action)
            add_video_db(command)
        return render_template('camera-survillance.html')
    else:
        return redirect(url_for('signin'))
    
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    if session['loggedin']:
        cmd='select videos from videos where user_id="{0}" order by created desc'.format(str(session['id']))
        vals=fetch_data(cmd)[0]['videos']
        print(vals)
        vals=vals.split('|')[0]
        return Response(gen(VideoCamera(vals)),mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return redirect('signin')
@app.route('/show_video')
def home():
    if session['loggedin']:
        LOG.info('Rendering home page')
        filename = request.args.get('filename','')
        print("*"*50)
        print("++"*10,filename)
        print("*"*50)
        response = render_template(
            'show_video.html',
            time=str(datetime.now()),
            video1=filename,
        )
        return response
    else:
        return redirect('signin')
@app.route('/list_video')
def list_vids():
    video_id = request.args.get('vid_id')
    print(video_id)
    vids=[]
    if session['loggedin']:
        if video_id==0 or video_id==None:
            cmd='select videos from videos where user_id="{0}" order by created desc'.format(str(session['id']))
            vals=fetch_data(cmd)[0]['videos']
            print(vals)
            vals=vals.split('|')
            vids.append([vals[0],vals[1],vals[2].split(',')])
            for i in range(3,len(vals),2):
                vids.append([vals[i],vals[i+1]])
            print(vids)
            LOG.info('Rendering home page')
            response = render_template(
               'processed.html',
                time=str(datetime.now()),
                vids=vids,
            )
            return response
        else:
            cmd='select videos from videos where user_id="{0}" and id={1}'.format(str(session['id']),video_id)
            vals=fetch_data(cmd)[0]['videos'].split('|')
            vids.append([vals[0],vals[1],vals[2].split(',')])
            for i in range(3,len(vals),2):
                vids.append([vals[i],vals[i+1]])
            print(vids)
            LOG.info('Rendering home page')
            response = render_template(
               'processed.html',
                time=str(datetime.now()),
                vids=vids,
            )
            return response
    else:
        return redirect('signin')
            

def partial_response(path, start, end=None):
    LOG.info('Requested: %s, %s', start, end)
    print(path)
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

@app.route('/video')
def video():
    if session['loggedin']:
        filename = request.args.get('filename')
        path = filename
        print(path)
    #    path = 'demo.mp4'

        start, end = get_range(request)
        return partial_response(path, start, end)
    else:
        return redirect('signin')
@app.route('/logout')
def logout():
    if session['loggedin']:
        session['loggedin']=False
        session['username']=None
        session['id']=None
    return redirect('signin')
if __name__ == '__main__':
    flask_options = dict(
        host='0.0.0.0',
        debug=True,
        port=5001,
        threaded=True,
    )
    logging.basicConfig(level=logging.INFO)
    app.run(**flask_options)
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(8080)
    IOLoop.instance().start()