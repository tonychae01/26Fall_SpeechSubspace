"""Small helpers shared by the notebooks and batch pilot."""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from nems import Model
from nems.layers import (WeightChannels, FiniteImpulseResponse,
                         RectifiedLinear, DoubleExponential)
from nems.tools.recording import load_recording
from nems.preprocessing.normalization import log_compress

ROOT = Path(__file__).resolve().parents[1]


def load_site():
    """Keep repeated stimuli in test; normalize using training data only."""
    path = next((ROOT / '01_raw_data/wingert2026').glob('CLT027c*.tgz'))
    rec = load_recording(str(path))
    stim, resp = rec['stim'].rasterize(), rec['resp'].rasterize()
    epochs = stim.epochs
    names = epochs.loc[epochs.name.str.startswith('STIM_'), 'name']
    counts = names.value_counts()

    def collect(selected):
        x, y = [], []
        for name in selected:
            x.append(stim.extract_epoch(name).mean(axis=0).T)
            y.append(resp.extract_epoch(name).mean(axis=0).T)
        return log_compress(np.stack(x)), np.stack(y)

    x_train, y_train = collect(sorted(counts[counts == 1].index))
    x_test, y_test = collect(sorted(counts[counts > 1].index))
    scales = {}
    for key, train, test in [('x', x_train, x_test), ('y', y_train, y_test)]:
        low = train.min(axis=(0, 1), keepdims=True)
        scale = train.max(axis=(0, 1), keepdims=True) - low
        scale[scale == 0] = 1
        train[:] = (train - low) / scale
        test[:] = (test - low) / scale
        scales[key + '_min'], scales[key + '_scale'] = low, scale
    return x_train, y_train, x_test, y_test, resp.chans, scales


def population_model(n_units, cnn=True, width=8):
    """Reduced-width pilot; width=80 uses 80/100/100 CNN hidden channels."""
    layers = [WeightChannels(shape=(32, 1, width)),
              FiniteImpulseResponse(shape=(15 if cnn else 25, 1, width))]
    if cnn:
        hidden = 100 if width == 80 else width
        layers += [RectifiedLinear(shape=(width,), no_offset=True),
                   WeightChannels(shape=(width, 1, hidden)),
                   FiniteImpulseResponse(shape=(10, 1, hidden)),
                   RectifiedLinear(shape=(hidden,), no_offset=True),
                   WeightChannels(shape=(hidden, hidden)),
                   RectifiedLinear(shape=(hidden,), no_offset=True)]
    else:
        hidden = width
    layers += [WeightChannels(shape=(hidden, n_units)),
               DoubleExponential(shape=(n_units,))]
    for layer in layers:
        if not layer.is_nonlinearity:
            layer.regularizer = 'l2:4'
    return Model(layers=layers, name='CNN' if cnn else 'LN')


def tuning_pca(filters, threshold=0.95):
    """Centered PCA across contexts; keep the mean filter separately."""
    flat = filters.reshape(len(filters), -1)
    pca = PCA(svd_solver='full').fit(flat)
    cumulative = pca.explained_variance_ratio_.cumsum()
    count = int(np.searchsorted(cumulative, threshold) + 1)
    return pca, cumulative, count


def prediction_correlation(prediction, response, warmup=25):
    """Exclude trial-boundary padding before pooling test trials."""
    p = prediction[:, warmup:].reshape(-1, prediction.shape[-1])
    y = response[:, warmup:].reshape(-1, response.shape[-1])
    return np.array([np.corrcoef(a, b)[0, 1] if a.std() and b.std()
                     else np.nan for a, b in zip(p.T, y.T)])


def tensorflow_model(model, history=25, channels=32):
    """Use NEMS layer conversions; fixed windows keep probing inexpensive."""
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(history, channels))
    output = inputs
    for layer in model.layers:
        output = layer.as_tensorflow_layer(input_shape=tuple(output.shape))(output)
    return tf.keras.Model(inputs, output)


def local_filters(network, stimulus, times, unit=0, history=25):
    """One scalar output gradient per context; result: context x frequency x lag."""
    import tensorflow as tf
    filters = []
    for t in times:
        window = tf.convert_to_tensor(stimulus[None, t-history+1:t+1], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(window)
            response = network(window, training=False)[0, -1, unit]
        filters.append(tape.gradient(response, window)[0].numpy().T)
    return np.stack(filters)
