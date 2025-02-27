#-*- coding:utf8  -*-

from __future__ import print_function

import random as pyrandom
import glob,sys,os,re,codecs,traceback
from pylab import *
from PIL import Image
from PIL import ImageFont,ImageDraw
from scipy.ndimage import filters,measurements,interpolation
from scipy.misc import imsave
import common
import chars

import argparse
parser = argparse.ArgumentParser(description = "Generate text line training data")
parser.add_argument('-o','--base',default='linegen',help='output directory, default: %(default)s')
parser.add_argument('-r','--distort',type=float,default=1.0)
parser.add_argument('-R','--dsigma',type=float,default=20.0)
parser.add_argument('-f','--fonts',default=None)
parser.add_argument('-F','--fontlist',default=None)
parser.add_argument('-t','--texts',default=None)
parser.add_argument('-T','--textlist',default=None)
parser.add_argument('-m','--maxlines',default=50000,type=int,
    help='max # lines for each directory, default: %(default)s')
parser.add_argument('-e','--degradations',default="lo",
    help="lo, med, or hi; or give a file, default: %(default)s")
parser.add_argument('-j','--jitter',default=0.5)
parser.add_argument('-s','--sizes',default="40-70")
parser.add_argument('-d','--display',action="store_true")
parser.add_argument('--numdir',action="store_true")
parser.add_argument('-C','--cleanup',default='[_~#]')
parser.add_argument('-D','--debug_show',default=None,
    help="select a class for stepping through")
args = parser.parse_args()

if "-" in args.sizes:
    lo,hi = args.sizes.split("-")
    sizes = range(int(lo),int(hi)+1)
else:
    sizes = [int(x) for x in args.sizes.split(",")]


if args.degradations=="lo":
    # sigma +/-   threshold +/-
    deglist = """
    0.5 0.0   0.5 0.0
    """
elif args.degradations=="med":
    deglist = """
    0.5 0.0   0.5 0.05
    1.0 0.3   0.4 0.05
    1.0 0.3   0.5 0.05
    1.0 0.3   0.6 0.05
    """
elif args.degradations=="hi":
    deglist = """
    0.5 0.0   0.5 0.0
    1.0 0.3   0.4 0.1
    1.0 0.3   0.5 0.1
    1.0 0.3   0.6 0.1
    1.3 0.3   0.4 0.1
    1.3 0.3   0.5 0.1
    1.3 0.3   0.6 0.1
    """
elif args.degradations is not None:
    with open(args.degradations) as stream:
        deglist = stream.read()

degradations = []
for deg in deglist.split("\n"):
    deg = deg.strip()
    if deg=="": continue
    deg = [float(x) for x in deg.split()]
    degradations.append(deg)

if args.fonts is not None:
    fonts = []
    for pat in args.fonts.split(':'):
        if pat=="": continue
        fonts += sorted(glob.glob(pat))
elif args.fontlist is not None:
    fonts = re.split(r'\s*\n\s*',open(args.fontlist).read())
else:
    print("use -f or -F arguments to specify fonts")
    sys.exit(1)
assert len(fonts)>0,"no fonts?"
print("fonts", fonts)

if args.texts is not None:
    texts = []
    for pat in args.texts.split(':'):
        print(pat)
        if pat=="": continue
        texts += sorted(glob.glob(pat))
elif args.textlist is not None:
    texts = re.split(r'\s*\n\s*',open(args.textlist).read())
else:
    print("use -t or -T arguments to specify texts")
    sys.exit(1)
assert len(texts)>0,"no texts?"

lines = []
for text in texts:
    print("# reading", text)
    with codecs.open(text,'r','utf-8') as stream:
        for line in stream.readlines():
            line = line.strip()
            line = re.sub(args.cleanup,'',line)
            if len(line)<1: continue
            lines.append(line)
print("got", len(lines), "lines")
assert len(lines)>0
lines = list(set(lines))
print("got", len(lines), "unique lines")

def rgeometry(image,eps=0.03,delta=0.3):
    m = array([[1+eps*randn(),0.0],[eps*randn(),1.0+eps*randn()]])
    w,h = image.shape
    c = array([w/2.0,h/2])
    d = c-dot(m,c)+array([randn()*delta,randn()*delta])
    return interpolation.affine_transform(image,m,offset=d,order=1,mode='constant',cval=image[0,0])

def rdistort(image,distort=3.0,dsigma=10.0,cval=0):
    h,w = image.shape
    hs = randn(h,w)
    ws = randn(h,w)
    hs = filters.gaussian_filter(hs,dsigma)
    ws = filters.gaussian_filter(ws,dsigma)
    hs *= distort/amax(hs)
    ws *= distort/amax(ws)
    def f(p):
        return (p[0]+hs[p[0],p[1]],p[1]+ws[p[0],p[1]])
    return interpolation.geometric_transform(image,f,output_shape=(h,w),
        order=1,mode='constant',cval=cval)

if args.debug_show:
    ion(); gray()

base = args.base
print("base", base)
os.system("rm -rf "+base)
os.mkdir(base)

def crop(image,pad=1):
    [[r,c]] = measurements.find_objects(array(image==0,'i'))
    r0 = r.start
    r1 = r.stop
    c0 = c.start
    c1 = c.stop
    image = image[r0-pad:r1+pad,c0-pad:c1+pad]
    return image

last_font = None
last_size = None
last_fontfile = None

def genline(text,fontfile=None,size=36,sigma=0.5,threshold=0.5):
    global image,draw,last_font,last_fontfile
    if last_fontfile!=fontfile or last_size!=size:
        last_font = ImageFont.truetype(fontfile,size)
        last_fontfile = fontfile
    font = last_font
    image = Image.new("L",(6000,200))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0,0,6000,6000),fill="white")
    # print("\t", size, font)
    draw.text((250,20),text,fill="black",font=font)
    a = asarray(image,'f')
    a = a*1.0/amax(a)
    if sigma>0.0:
        a = filters.gaussian_filter(a,sigma)
    a += clip(randn(*a.shape)*0.2,-0.25,0.25)
    a = rgeometry(a)
    a = array(a>threshold,'f')
    a = crop(a,pad=3)
    # FIXME add grid warping here
    # clf(); ion(); gray(); imshow(a); ginput(1,0.1)
    del draw
    del image
    return a

lines_per_size = args.maxlines//len(sizes)
for pageno,font in enumerate(fonts):
    if args.numdir:
        pagedir = "%s/%04d"%(base,pageno+1)
    else:
        fbase = re.sub(r'^[./]*','',font)
        fbase = re.sub(r'[.][^/]*$','',fbase)
        fbase = re.sub(r'[/]','_',fbase)
        pagedir = "%s/%s"%(base,fbase)
    os.mkdir(pagedir)
    print("===", pagedir, font)
    lineno = 0
    while lineno<args.maxlines:
        (sigma,ssigma,threshold,sthreshold) = pyrandom.choice(degradations)
        sigma += (2*rand()-1)*ssigma
        threshold += (2*rand()-1)*sthreshold
        line = pyrandom.choice(lines)
        size = pyrandom.choice(sizes)
        with open(pagedir+".info","w") as stream:
            stream.write("%s\n"%font)
        try:
            image = genline(text=line,fontfile=font,
                size=size,sigma=sigma,threshold=threshold)
        except:
            traceback.print_exc()
            continue
        if amin(image.shape)<10: continue
        if amax(image)<0.5: continue
        if args.distort>0:
            image = rdistort(image,args.distort,args.dsigma,cval=amax(image))
        if args.display:
            gray()
            clf(); imshow(image); ginput(1,0.1)
        fname = pagedir+"/01%04d"%lineno
        imsave(fname+".bin.png",image)
        gt = common.normalize_text(line)
        with codecs.open(fname+".txt","w",'utf-8') as stream:
            stream.write(gt+"\n")
        print("%5.2f %5.2f %3d\t%s" % (sigma, threshold, size, line))
        lineno += 1
