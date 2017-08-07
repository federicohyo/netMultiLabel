# Author Federico Corradi
# federico.corradi@gmail.com
# August 2017
# create hdf5 dataset with multi label 

import numpy as np
import os
import matplotlib.image as img
import scipy.misc
import caffe
import h5py

#parameters
img_dir = "img_label/"
img_saves = "img_single/"
threshold_ok = 800000 
space_resolution = 20
train_perc = 0.8
test_perc = 0.2
#caffe
dimensions = 3
image_size_x = 346
image_size_y = 260
shuffle = True
regenerateImages = False
#remove previous files
os.system("rm training_set.txt")
os.system("rm testing_set.txt")
os.system("rm test.h5")
os.system("rm train.h5")

#read label files
all_labels_files = []
for file in os.listdir("."):
    if file.endswith(".txt"):
        all_labels_files.append(file)

labels = []
for this_file in all_labels_files:
    labels.append([line.rstrip('\n') for line in open(this_file)])

flat_labels = [item for sublist in labels for item in sublist]

#unique
flat_labels_un = list(set(flat_labels))

#xtot
xtot = []
ytot = []
ztot = []
img_names = []
#make file labels and check for image 
for this_label in flat_labels_un:
    xt, yt, zt, tt, nt = this_label.split(",")
    key = tt.split("t: ")[1]
    img_name = 'img__'+key+'.png'
    namefile = img_dir+img_name
    namefiles = img_saves+img_name
    if(os.path.isfile(namefile)):
        #file exists, now load file and check it's number of points
        image = img.imread(namefile)
        image = image[:,:,1] #only green channel
        image *= (255.0/image.max())#normalize
        if(np.sum(image) > threshold_ok):
            scipy.misc.imsave(namefiles, image)
            #add labels 
            xtot.append(float(xt.split('x:')[1]))
            ytot.append(float(yt.split('y:')[1]))
            ztot.append(float(zt.split('z:')[1]))
            img_names.append(img_name)
        else:
            print("not enough events in image "+namefile)


xtot = np.array(xtot)
ytot = np.array(ytot)
ztot = np.array(ztot)
label_interval = np.linspace(-1,1,space_resolution)

def find_closest(label_interval, value):
    return int(np.where(label_interval == min(label_interval, key=lambda x:abs(x-value)))[0])

#hot encode labels
xlab = []
ylab = []
zlab = []
for i in range(len(xtot)):
    xlab.append(np.eye(space_resolution)[find_closest(label_interval,xtot[i])].astype(int))
    ylab.append(np.eye(space_resolution)[find_closest(label_interval,ytot[i])].astype(int))
    zlab.append(np.eye(space_resolution)[find_closest(label_interval,ztot[i])].astype(int))


#make caffe train file
train_num = int(np.floor(len(xlab)*train_perc))
test_num = int(np.floor(len(xlab)*test_perc))
num_examples = train_num+test_num
indexes = np.linspace(0,num_examples,num_examples+1).astype('int') 
if(shuffle):
    np.random.shuffle(indexes)   

ff = open("training_set.txt", "w+")
for i in range(train_num):
    ff.write(img_names[indexes[i]]+' '+str(xlab[indexes[i]]).strip("[ ]")+' '+str(ylab[indexes[i]]).strip("[ ]")+' '+str(zlab[indexes[i]]).strip("[ ]")+'\n')
ff.close()

ff = open("testing_set.txt", "w+")
for i in range(train_num,train_num+test_num):
    ff.write(img_names[indexes[i]]+' '+str(xlab[indexes[i]]).strip("[ ]")+' '+str(ylab[indexes[i]]).strip("[ ]")+' '+str(zlab[indexes[i]]).strip("[ ]")+'\n')
ff.close()

## MAKE hdf5
f = h5py.File('train.h5', 'w')
# 1200 data, each is a 128-dim vector
f.create_dataset('data', (train_num, image_size_x*image_size_y), dtype='int')
# Data's labels, each is a 4-dim vector
f.create_dataset('label', (train_num, space_resolution*dimensions), dtype='int')
# Fill in something with fixed pattern
# Regularize values to between 0 and 1, or SigmoidCrossEntropyLoss will not work
print("creating train hdf5 files...")
with open( 'training_set.txt', 'r' ) as T :
    lines = T.readlines()
for i,l in enumerate(lines):
    if(i < train_num):
        sp = l.split(' ')
        img = caffe.io.load_image( img_saves+sp[0] , color=False)
        #img = caffe.io.resize( img, (image_size, image_size, 1) ) # resize to fixed size
        f['data'][i] = np.reshape(img,[image_size_x*image_size_y])
        tmpl = np.append(xlab[i],ylab[i])
        tmpl = np.append(tmpl,zlab[i])
        f['label'][i] = tmpl
    else:
        break
f.close()
print("done")


f = h5py.File('test.h5', 'w')
# 1200 data, each is a 128-dim vector
f.create_dataset('data', (test_num, image_size_x*image_size_y), dtype='int')
# Data's labels, each is a 4-dim vector
f.create_dataset('label', (test_num, space_resolution*dimensions), dtype='int')
# Fill in something with fixed pattern
# Regularize values to between 0 and 1, or SigmoidCrossEntropyLoss will not work
print("creating test hdf5 files...")
with open( 'testing_set.txt', 'r' ) as T :
    lines = T.readlines()
for i,l in enumerate(lines):
    sp = l.split(' ')
    img = caffe.io.load_image( img_saves+sp[0] , color=False)
    #img = caffe.io.resize( img, (image_size, image_size, 1) ) # resize to fixed size
    f['data'][i] = np.reshape(img,[image_size_x*image_size_y])
    tmpl = np.append(xlab[i],ylab[i])
    tmpl = np.append(tmpl,zlab[i])
    f['label'][i] = tmpl
f.close()
print("done")

#h5f = h5py.File('test.h5','r')
#h5f.close()

