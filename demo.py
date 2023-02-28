import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import timeit
#指定在cpu上运行
def cpu_run():
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([10000, 1000])
        cpu_b = tf.random.normal([1000, 2000])
        cpu_c = tf.matmul(cpu_a, cpu_b)
        # print( "cpu_a: ", cpu_a.device)
        # print( "cpu_b: ", cpu_b.device)
        # print("cpu_c:", cpu_c.device)
    return cpu_c

#指定在gpu上运行

def gpu_run():
    with tf.device( '/gpu:0'):
        gpu_a = tf.random. normal([ 10000,1000])
        gpu_b = tf.random. normal([ 1000, 2000])
        gpu_c = tf.matmul(gpu_a, gpu_b)
        # print( "gpu_a: ", gpu_a.device)
        # print("gpu_b: ", gpu_b.device)
        # print("gpu_c: ", gpu_c.device)
    return gpu_c

cpu_time = timeit.timeit(cpu_run, number = 10)
gpu_time = timeit.timeit(gpu_run, number = 10)
print('cpu:',cpu_time, 'gpu:',gpu_time)
