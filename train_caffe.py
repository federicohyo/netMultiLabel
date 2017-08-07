import os, sys

PROJECT_HOME = '/Users/federicocorradi/Documents/of_v0.9.8_osx_release/addons/ofxSiliconRetina/example_siliconretinaGuiAruco/bin/data/training/'
CAFFE_HOME = '/Volumes/128GB/inilabs/caffe_git/'
os.chdir(PROJECT_HOME)

sys.path.insert(0, CAFFE_HOME + 'caffe/python')
import caffe, h5py

from pylab import *
from caffe import layers as L

def net(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=20, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.ip2, in_place=True)
    n.ip3 = L.InnerProduct(n.relu2, num_output=60, weight_filler=dict(type='xavier'))
    n.loss = L.SigmoidCrossEntropyLoss(n.ip3, n.label)
    return n.to_proto()

with open(PROJECT_HOME + 'auto_train.prototxt', 'w') as f:
    f.write(str(net(PROJECT_HOME + 'train.h5list', 50)))
with open(PROJECT_HOME + 'auto_test.prototxt', 'w') as f:
    f.write(str(net(PROJECT_HOME + 'test.h5list', 20)))

caffe.set_mode_cpu()
solver = caffe.SGDSolver(PROJECT_HOME + 'auto_solver.prototxt')

solver.net.forward()
solver.test_nets[0].forward()
solver.step(1)

niter = 200
test_interval = 10
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter * 1.0 / test_interval)))
print len(test_acc)
output = zeros((niter, 20, 60))

# The main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='data')
    output[it] = solver.test_nets[0].blobs['ip3'].data[:60]

    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        data = solver.test_nets[0].blobs['ip3'].data
        label = solver.test_nets[0].blobs['label'].data
        for test_it in range(100):
            solver.test_nets[0].forward()
            # Positive values map to label 1, while negative values map to label 0
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if data[i][j] > 0 and label[i][j] == 1:
                        correct += 1
                    elif data[i][j] <= 0 and label[i][j] == 0:
                        correct += 1
        test_acc[int(it / test_interval)] = correct * 1.0 / (len(data) * len(data[0]) * 100)

# Train and test done, outputing convege graph
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
_.savefig('converge.png')

# Check the result of last batch
print solver.test_nets[0].blobs['ip3'].data
print solver.test_nets[0].blobs['label'].data

