"""Verify GPU inference and gradients inside an allocated compute-node job."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('catscan'):
    raise SystemExit('Run this check inside a Slurm compute-node allocation.')
subprocess.run(['bash', '-c', 'source ~/.config/cluster/shell.sh; gpu_here'], check=True)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
from strf import population_model, tensorflow_model, local_filters, ROOT

assert tf.config.list_physical_devices('GPU'), 'TensorFlow cannot see the allocated GPU.'
for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
model = population_model(20).sample_from_priors()
network = tensorflow_model(model)
x = tf.constant(np.random.default_rng(0).uniform(size=(1, 25, 32)), dtype=tf.float32)
with tf.GradientTape() as tape:
    response = network(x)
    loss = tf.reduce_mean(response ** 2)
gradients = tape.gradient(loss, network.trainable_variables)
assert 'GPU' in response.device
assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
d = local_filters(network, x.numpy()[0], [24])
assert np.isfinite(d).all()
result = dict(job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
              response_device=response.device, parameter_gradients_finite=True,
              input_gradients_finite=True, tensorflow=tf.__version__)
path = ROOT / '05_results/wingert_smoke/gpu_check.json'
path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
