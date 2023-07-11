import os
import json

import tensorflow as tf
import classification

strategy = tf.distribute.MultiWorkerMirroredStrategy()

per_worker_batch_size = 32
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
})
num_workers = 2


gloabal_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = classification.flower_dataset(gloabal_batch_size)

with strategy.scope():
    multi_worker_model = classification.build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=20)