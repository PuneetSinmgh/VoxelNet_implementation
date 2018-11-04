#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

from config import cfg


class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = int(out_channels / 2)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, mask, training):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)

        #n [K, 1, units]
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)

        # [K, T, units]
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])

        # [K, T, 2 * units]
        concatenated = tf.concat([pointwise, repeated], axis=2)

        mask = tf.tile(mask, [1, 1, 2 * self.units])

        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32))

        return concatenated


class FeatureNet(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet, self).__init__()
        self.training = training

        # scalar
        self.batch_size = batch_size
        # [ΣK, 35/45, 7]
        self.feature = tf.placeholder(
            tf.float32, [None, cfg.VOXEL_POINT_COUNT, 7], name='feature')
        # [ΣK]
        self.number = tf.placeholder(tf.int64, [None], name='number')
        # [ΣK, 4], each row stores (batch, d, h, w)
        self.coordinate = tf.placeholder(
            tf.int64, [None, 4], name='coordinate')

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.vfe1 = VFELayer(32, 'VFE-1')
            self.vfe2 = VFELayer(128, 'VFE-2')
            
# FCN layer was missing, as per the paper, it concist of a relu , linear , and a batch normalization layer

            #A Linear relu net which takes input from VFE layers 
            self.linear-relunet = tf.layers.Dense(
                128, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        # boolean mask [K, T, 2 * units]
        mask = tf.not_equal(tf.reduce_max(
            self.feature, axis=2, keep_dims=True), 0)
        x = self.vfe1.apply(self.feature, mask, self.training)
        x = self.vfe2.apply(x, mask, self.training)

# point wise concatinated feature as input to relu layer and batch normalization 
        x= self.linear-relunet.apply(x)
        x= self.batch_norm(x,self.training)
        

        # [ΣK, 128]
        #element wise max pooling of feature 
        voxelwise = tf.reduce_max(x, axis=1)

        # car: [N * 10 * 400 * 352 * 128]
        # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
        #sparse 4-D tensors : represents point wise concatenated feature
        self.outputs = tf.scatter_nd(
            self.coordinate, voxelwise, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])



def build_input(voxel_dictionary):
    batch_size = len(voxel_dictionary)

    feature_list = []
    number_list = []
    coordinate_list = []
    
   # extracts feature buffer , number of points in a voxel , and co-ordinate buffer  
    for i, voxel_dict in zip(range(batch_size), voxel_dictionary):
        feature_list.append(voxel_dict['feature_buffer'])
        number_list.append(voxel_dict['number_buffer'])
        coordinate = voxel_dict['coordinate_buffer']
        coordinate_list.append(
            np.pad(coordinate, ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

    feature = np.concatenate(feature_list)
    number = np.concatenate(number_list)
    coordinate = np.concatenate(coordinate_list)
    return batch_size, feature, number, coordinate


#run the Feature learning network over the GPU and prints the results back 
def run(batch_size, feature, number, coordinate):

    #setting up gpu optiond to run on 
    gpu_options = tf.GPUOptions(visible_device_list='1')
    
    
    with tf.Session(config=tf.ConfigProto(tf.ConfigProto(
        gpu_options=gpu_options,
        device_count={'GPU': 1}))) as session:
        
        #Setting up the Feature Learning Network
        model = FeatureNet(training=False, batch_size=batch_size)
        
        tf.global_variables_initializer().run()
        for i in range(10):
            feed = {model.feature: feature,
                    model.number: number,
                    model.coordinate: coordinate}
            outputs = session.run([model.outputs], feed)
            print(outputs[0].shape)



def main():
    
    #path to the Train , Test , Kitti validation Data
    data_dir = './data/object/training/voxel'
    batch_size = 32

    flist = [f for f in os.listdir(data_dir) if f.endswith('npz')]

    #simple dictionary to hold the  data files upto range of batch_size
    voxel_dictionary = []
    for id in range(0, len(flist), batch_size):
        pre_time = time.time()
        batch_file = [f for f in flist[id:id + batch_size]]
        voxel_dictionary = []
        for file in batch_file:
            voxel_dictionary.append(np.load(os.path.join(data_dir, file)))

        # example input with batch size 16
        batch_size, feature, number, coordinate = build_input(voxel_dictionary)
        print(time.time() - pre_time)
#transforms each raw cloud point, in a batch size of the cloud points, into a vector representation charachterizing the shape informantion 
    run(batch_size, feature, number, coordinate)


if __name__ == '__main__':
    main()
