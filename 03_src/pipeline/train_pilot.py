"""One small LN/CNN fit. Run through Slurm, never on the login node."""
import argparse
import json
import os
import socket
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/speech-mpl')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='04_checkpoints/wingert_pilot_seed0')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('catscan'):
        parser.error('Training requires a Slurm compute-node allocation.')
    if args.epochs < 2 or args.width < 1:
        parser.error('Use at least two epochs and a positive width.')

    import subprocess
    subprocess.run(['bash', '-c', 'source ~/.config/cluster/shell.sh; gpu_here'], check=True)
    import numpy as np
    import tensorflow as tf
    from nems.tools.json import save_model
    from strf import load_site, population_model, prediction_correlation, tuning_pca, tensorflow_model, local_filters, ROOT

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)  # Avoid overwriting an earlier run.
    tf.keras.utils.set_random_seed(args.seed)
    x, y, xt, yt, cells, scales = load_site()
    np.savez(output / 'normalization.npz', **scales)
    metrics = {'settings': vars(args), 'site': 'CLT027c', 'cellids': cells,
               'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
               'tensorflow_gpus': [v.name for v in tf.config.list_physical_devices('GPU')],
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'train_shape': list(x.shape), 'test_shape': list(xt.shape),
               'scope': 'Small single-fit pilot; no jackknife or noise correction.'}
    models = {}
    for name, cnn in [('LN', False), ('CNN', True)]:
        print(f'Starting {name}', flush=True)
        model = population_model(len(cells), cnn=cnn, width=args.width).sample_from_priors()
        for stage, rate in [(1, 0.01), (2, 0.001)]:
            if stage == 1:
                model.layers[-1].skip_nonlinearity()
            else:
                model.layers[-1].unskip_nonlinearity()
            model = model.fit(x, y, backend='tf', batch_size=None, verbose=0,
                              fitter_options={'cost_function': 'squared_error',
                                              'epochs': args.epochs // 2,
                                              'learning_rate': rate,
                                              'validation_split': 0})
        print(f'Finished {name}', flush=True)
        save_model(model, str(output / f'{name}.json'))
        pred = model.predict(xt, batch_size=len(xt))
        r = prediction_correlation(pred, yt)
        metrics[name + '_test_r'] = [float(v) if np.isfinite(v) else None for v in r]
        models[name] = model
        if name == 'LN':
            spectral = model.layers[0].coefficients[:, 0, :]
            temporal = model.layers[1].coefficients[:, 0, :]
            readout = model.layers[2].coefficients
            h = np.einsum('fr,tr,ru->uft', spectral, temporal, readout)
            np.save(output / 'LN_strfs.npy', h[:, :, ::-1])

    # Probe a preselected unit on one training trial, without crossing boundaries.
    times = np.linspace(25, x.shape[1] - 1, 64, dtype=int)
    d = local_filters(tensorflow_model(models['CNN']), x[0], times)
    pca, cumulative, k = tuning_pca(d)
    np.savez(output / 'unit0_probe.npz', dstrf=d, times=times,
             mean_filter=d.mean(axis=0), components=pca.components_, cumulative=cumulative)
    metrics['unit0_dstrf_k95'] = k
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
