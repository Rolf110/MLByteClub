from numpy import (
    array,
    zeros,
    dot,
    median,
    nan,
    reshape,
    log2,
    linspace,
    argmin,
    abs,
    delete,
    inf,
    argmax,
    prod,
    zeros_like,
    load,
)
from math import ceil, floor
from scipy.linalg import norm
from tensorflow import convert_to_tensor
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import function as Kfunction
from tensorflow.keras.models import Model, clone_model
from tensorflow.image import extract_patches
from typing import List, Generator
from collections import namedtuple
from itertools import product
from time import time
import concurrent.futures
import h5py
import os
import gc
from glob import glob

# Define namedtuples for more interpretable return types.
SegmentedData = namedtuple("SegmentedData", ["wX_seg", "qX_seg"])

# Define static functions to use for multiprocessing
def _bit_round_parallel(t: float, alphabet: array) -> float:
    """Rounds a quantity to the nearest atom in the (scaled) quantization alphabet.

    Parameters
    -----------
    t : float
        The value to quantize.
    alphabet : array
        Scalar quantization alphabet.

    Returns
    -------
    bit : float
        The quantized value.
    """

    # Scale the alphabet appropriately.
    return alphabet[argmin(abs(alphabet - t))]

def _quantize_weight_parallel(
    w: float, u: array, X: array, X_tilde: array, alphabet: array
) -> float:
    """Quantizes a single weight of a neuron.

    Parameters
    -----------
    w : float
        The weight.
    u : array ,
        Residual vector.
    X : array
        Vector from the analog network's random walk.
    X_tilde : array
        Vector from the quantized network's random walk.
    alphabet : array
        Scalar quantization alphabet.

    Returns
    -------
    bit : float
        The quantized value.
    """

    if norm(X_tilde, 2) < 10 ** (-16):
        return 0

    if abs(dot(X_tilde, u)) < 10 ** (-10):
        return _bit_round_parallel(w, alphabet)

    return _bit_round_parallel(dot(X_tilde, u + w * X) / (norm(X_tilde, 2) ** 2), alphabet)

def _quantize_neuron_parallel(
    w: array,
    hf_filename: str,
    alphabet: array,
) -> array:
    """Quantizes a single neuron in a Dense layer.

    Parameters
    -----------
    w: array
        The neuron to be quantized.
    hf_filename: str
        Filename for hdf5 file with datasets wX, qX, transposed.
    alphabet : array
        Scalar quantization alphabet

    Returns
    -------
    q: array
        Quantized neuron.
    """

    with h5py.File(hf_filename, 'r') as hf:
        N_ell = hf['wX'].shape[0]
        u = zeros(hf['wX'].shape[1])
        q = zeros(N_ell)
        for t in range(N_ell):
            q[t] = _quantize_weight_parallel(w[t], u, hf['wX'][t, :], hf['qX'][t, :], alphabet)
            u += w[t] * hf['wX'][t, :] - q[t] * hf['qX'][t, :]

    return q


class Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        """Constructs a child class of the Keras Sequence class to generate batches
        of images. 

        Parameters
        -----------
        x_set : 1D-array
            Array of images
        y_set : 1D-array
            Labels for the images.
        batch_size: int
            Specifies how many images to generate in a batch.
        """

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return array(batch_x), array(batch_y)

class QuantizedNeuralNetwork:
    def __init__(
        self,
        network: Model,
        batch_size: int,
        get_data: Generator[array, None, None],
        mini_batch_size=32,
        logger=None,
        ignore_layers=[],
        bits=log2(3),
        alphabet_scalar=1,
    ):
        """This is a wrapper class for a tensorflow.keras.models.Model class
        which handles quantizing the weights for Dense layers.

        Parameters
        -----------
        network : Model
            The pretrained neural network.
        batch_size : int,
            How many training examples to use for learning the quantized weights in a
            given layer.
        get_data : Generator
            A generator for yielding training examples for learning the quantized weights.
        mini_batch_size: int
            How many training examples to feed through the hidden layers at a time. We can't feed
            in the entire batch all at once if the batch is large since CPUs and GPUs are memory 
            constrained.
        logger : logger
            A logging object to write updates to. If None, updates are written to stdout.
        ignore_layers : list of ints
            A list of layer indices to indicate which layers are *not* to be quantized.
        bits : float
            How many bits to use for the quantization alphabet. There are 2**bits characters
            in the quantization alphabet.
        alphabet_scalar : float
            A scaling parameter used to adjust the radius of the quantization alphabets for
            each layer.
        """

        self.get_data = get_data

        # The pre-trained network.
        self.trained_net = network

        # This copies the network structure but not the weights.
        self.quantized_net = clone_model(network)

        # Set all the weights to be the same a priori.
        self.quantized_net.set_weights(network.get_weights())

        self.alphabet_scalar = alphabet_scalar

        # Create a dictionary encoding which layers are Dense, and what their dimensions are.
        self.layer_dims = {
            layer_idx: layer.get_weights()[0].shape
            for layer_idx, layer in enumerate(network.layers)
            if layer.__class__.__name__ == "Dense"
        }

        # This determines the alphabet. There will be 2**bits atoms in our alphabet.
        self.bits = bits

        # Construct the (unscaled) alphabet. Layers will scale this alphabet based on the
        # distribution of that layer's weights.
        self.alphabet = linspace(-1, 1, num=int(round(2 ** (bits))))

        self.logger = logger

        self.ignore_layers = ignore_layers

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _get_layer_data_generator(self, layer_idx: int, transpose=False):
        """Gets the input data for the layer at a given index.

        Parameters
        -----------
        layer_idx : int
            Index of the layer.
        transpose: bool
            Whether to transpose the hidden activations or not. This is convenient for quantizing
            Dense layers when reading the hidden feature datas as rows will speed up i/o from hdf5 file.

        Returns
        -------
        hf_filename : str
            Filename of hdf5 file that contains datasets wX, qX.
        """

        # Determine how many inbound layers there are.
        if layer_idx == 0:
            # Don't need to feed data through hidden layers.
            inbound_analog_layers = None
            inbound_quant_layers = None
        else:
            # Determine whether there is more than one input layer. 
            inbound_analog_nodes = self.trained_net.layers[layer_idx].inbound_nodes
            if len(inbound_analog_nodes) > 1:
                self._log(f"Number of inbound analog nodes = {inbound_analog_nodes}...not sure what to do here!")
            else:
                inbound_analog_layers = inbound_analog_nodes[0].inbound_layers

            inbound_quant_nodes = self.quantized_net.layers[layer_idx].inbound_nodes
            if len(inbound_quant_nodes) > 1:
                self._log(f"Number of inbound quantized nodes = {inbound_quant_nodes}...not sure what to do here!")
            else:
                inbound_quant_layers = inbound_quant_nodes[0].inbound_layers

            # Sanity check that the two networks have the same number of inbound layers
            try:
                assert(len(inbound_analog_layers) == len(inbound_quant_layers))
            except TypeError: 
                # inbound_*_layers is a layer object, not a list
                inbound_analog_layers = [inbound_analog_layers]
                inbound_quant_layers = [inbound_quant_layers]


        layer = self.trained_net.layers[layer_idx]
        layer_data_shape = layer.input_shape[1:] if layer.input_shape[0] is None else layer.input_shape
        num_inbound_layers = len(inbound_analog_layers) if layer_idx != 0 else 1
        input_analog_layer = self.trained_net.layers[0]
        # Prebuild the partial models, since they require retracing so you don't want to do them inside a loop.
        if layer_idx > 0:
            prev_trained_model = Model(inputs=input_analog_layer.input,
                                     outputs=[analog_layer.output for analog_layer in inbound_analog_layers])
            prev_quant_model = Model(inputs=self.quantized_net.layers[0].input,
                     outputs=[quant_layer.output for quant_layer in inbound_quant_layers])

        # Preallocate space for h5py file.
        # NOTE:  # Technically num_images is an upper bound, since not every batch has full batch size.
        # I'm implicitly assuming that h5py zerofills pre-allocated space.
        num_images = self.get_data.__len__()*self.get_data.batch_size
        hf_dataset_shape = (num_inbound_layers*num_images, *layer_data_shape)
        if transpose:
            hf_dataset_shape = hf_dataset_shape[::-1]
        hf_filename = f"layer{layer_idx}_data.h5"

        with h5py.File(hf_filename, 'w') as hf:
            for batch_idx in range(self.get_data.__len__()):
                # Grab the batch of data, ignoring labels.
                mini_batch = self.get_data.__getitem__(batch_idx)[0]

                if layer_idx == 0:
                    # No hidden layers to pass data through.
                    wX = mini_batch
                    qX = mini_batch
                else:
                    wX = prev_trained_model.predict_on_batch(mini_batch)
                    qX = prev_quant_model.predict_on_batch(mini_batch)

                if batch_idx == 0:
                    hf.create_dataset("wX", shape=hf_dataset_shape)
                    hf.create_dataset("qX", shape=hf_dataset_shape)

                if transpose:
                    hf["wX"][..., batch_idx*wX.shape[0]:(batch_idx+1)*wX.shape[0]] = wX.T
                    hf["qX"][..., batch_idx*qX.shape[0]:(batch_idx+1)*qX.shape[0]] = qX.T
                else:
                    hf["wX"][batch_idx*wX.shape[0]:(batch_idx+1)*wX.shape[0]] = wX
                    hf["qX"][batch_idx*qX.shape[0]:(batch_idx+1)*qX.shape[0]] = qX


                # Dereference and call garbage collection--just to be safe--to free up memory.
                del mini_batch, wX, qX
                gc.collect()

        return hf_filename

    def _update_weights(self, layer_idx: int, Q: array):
        """Updates the weights of the quantized neural network given a layer index and
        quantized weights.

        Parameters
        -----------
        layer_idx : int
            Index of the Conv2D layer.
        Q : array
            The quantized weights.
        """

        # Update the quantized network. Use the same bias vector as in the analog network for now.
        if self.trained_net.layers[layer_idx].use_bias:
            bias = self.trained_net.layers[layer_idx].get_weights()[1]
            self.quantized_net.layers[layer_idx].set_weights([Q, bias])
        else:
            self.quantized_net.layers[layer_idx].set_weights([Q])

    def _quantize_layer_parallel(self, layer_idx: int):
        """Quantizes a Dense layer of a multi-layer perceptron.

        Parameters
        -----------
        layer_idx : int
            Index of the Dense layer.
        """

        W = self.trained_net.layers[layer_idx].get_weights()[0]
        N_ell, N_ell_plus_1 = W.shape
        # Placeholder for the weight matrix in the quantized network.
        Q = zeros(W.shape)
        N_ell_plus_1 = W.shape[1]

        self._log("\tFeeding input data through hidden layers...")
        tic = time()
        hf_filename = self._get_layer_data_generator(layer_idx, transpose=True)
        self._log(f"\tdone. {time()-tic:2f} seconds.")

        # Set the radius of the alphabet.
        rad = self.alphabet_scalar * median(abs(W.flatten()))
        layer_alphabet = rad*self.alphabet

        self._log("\tQuantizing neurons (in parallel)...")
        tic = time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Build a dictionary with (key, value) = (q, neuron_idx). This will
            # help us map quantized neurons to the correct neuron index as we call
            # _quantize_neuron asynchronously.
            future_to_neuron = {executor.submit(_quantize_neuron_parallel, W[:, neuron_idx], 
                hf_filename,
                layer_alphabet,
                ): neuron_idx for neuron_idx in range(N_ell_plus_1)}
            for future in concurrent.futures.as_completed(future_to_neuron):
                neuron_idx = future_to_neuron[future]
                try:
                    # Populate the appropriate column in the quantized weight matrix
                    # with the quantized neuron
                    Q[:, neuron_idx] = future.result()
                except Exception as exc:
                    self._log(f'\t\tNeuron {neuron_idx} generated an exception: {exc}')
                    raise exc

                self._log(f'\t\tNeuron {neuron_idx} of {N_ell_plus_1} quantized successfully.')

            # Set the weights for the quantized network.
            self._update_weights(layer_idx, Q)
        self._log(f"\tdone. {time()-tic:.2f} seconds.")

        # Now delete the hdf5 file.
        os.remove(f"./{hf_filename}")

    def quantize_network(self):
        """Quantizes all Dense layers that are not specified by the list of ignored layers."""

        # This must be done sequentially.
        num_layers = len(self.trained_net.layers)
        for layer_idx, layer in enumerate(self.trained_net.layers):
            if (
                layer.__class__.__name__ == "Dense"
                and layer_idx not in self.ignore_layers
            ):
                # Only quantize dense layers.
                tic = time()
                self._log(f"Quantizing layer {layer_idx} (in parallel) of {num_layers}...")
                self._quantize_layer_parallel(layer_idx)
                self._log(f"Layer {layer_idx} of {num_layers} quantized successfully in {time() - tic:.2f} seconds.")