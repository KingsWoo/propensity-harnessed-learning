import numpy as np
import tensorflow as tf

from tensorflow import keras 
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.regularizers import L1L2
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras.engine.base_layer import Layer

epsilon = 1e-7

# ----------------------- validation setup ----------------------- 
# divide physicians into n_folds groups
def validation_setup(n_folds, n_providers_per_fold, seed=0):

    n_providers = n_folds * n_providers_per_fold
    np.random.seed(seed)
    rndperm = np.random.permutation(range(n_providers))
    fold_groups = np.reshape(rndperm, [n_folds, n_providers_per_fold])

    # build valid_list, train_list
    valid_list = fold_groups
    train_list = []

    # in each fold of validation, this fold of physicians are used as validation set, while the other n_folds-1 folds are divided and used as two seperated training sets
    for i_fold in range(n_folds):

        np.random.seed(seed=i_fold+seed)
        rndperm = np.random.permutation(n_folds)
        rndperm = [_x for _x in rndperm if _x != i_fold]
        train_list.append([fold_groups[rndperm[:n_folds//2]].flatten(), fold_groups[rndperm[n_folds//2:]].flatten()])

    return train_list, valid_list

# ------------------------- custom losses -------------------------
# anchor loss
def AnchorLoss(y_true, y_pred):
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    
    loss = y_true * tf.math.log(y_true / y_pred) + (1-y_true) * tf.math.log((1-y_true) / (1-y_pred))
    
    return loss
    
# WCE loss
def WCELoss(y_true, y_pred, e=1):

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
           
    l1 = -tf.math.log(y_pred)
    l0 = -tf.math.log(1-y_pred)
        
    loss = y_true * (1/e*l1+(1-1/e)*l0) + (1-y_true) * l0
    
    return loss

# ------------------------- custom layers -------------------------
# multi-head dense, which is used as the K-branches in the G model
class MultiheadDense(Layer):

    def __init__(self, 
                 units=32, 
                 heads=1,
                 positional_mask=False,
                 activation='relu',
                 kernel_regularizer=None,
                 bias_regularizer=None, 
                 **kargs):

        super(MultiheadDense, self).__init__(**kargs)
        self.units = units
        self.heads = heads
        self.positional_mask = positional_mask
        self.kernel_initializer = initializers.get('glorot_uniform')
        self.bias_initializer = initializers.get('zeros')
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation = activations.get(activation)

    def build(self, input_shape):  

        self.kernel = self.add_weight(shape=(input_shape[-1], self.heads, self.units), 
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer)

        self.bias = self.add_weight(shape=(self.heads, self.units),
                                    name='bias',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer)

        # prevent using the disease probablity to affect the division of f and gk
        if self.positional_mask:

            self.kernel = tf.einsum('jhk, jk->jhk', self.kernel, 1 - tf.eye(self.units))

        self.built = True

    def call(self, inputs):

        if len(inputs.shape) == 2:
            einsum_format = 'ij,jhk->ihk'
        elif len(inputs.shape) == 3:
            einsum_format = 'ihj,jhk->ihk'

        outputs = tf.einsum(einsum_format, inputs, self.kernel)
        outputs = K.bias_add(outputs, self.bias)
        outputs = self.activation(outputs)

        return outputs    
    
# ------------------------- packaged models -------------------------

def MLP_model(params):
    
    n_inputs = params['n_inputs']
    n_labels = params['n_labels']
    layer_shape = params['layer_shape']
    common_reg = params['common_reg']
    
    # inputs
    x_input = keras.Input(shape=(n_inputs,), name='x_inputs')
    y_input = keras.Input(shape=(n_labels,), name='y_inputs')
    
    common_layer = x_input
    
    # MLP layers
    for i_layer, this_layer_shape in enumerate(layer_shape):
        
        common_layer = layers.Dense(this_layer_shape, 
                                    activation='relu', 
                                    kernel_regularizer=L1L2(l1=common_reg, l2=common_reg), 
                                    bias_regularizer=L1L2(l1=common_reg, l2=common_reg), 
                                    name='dense_layer_%d' % i_layer
                                   )(common_layer)
    
    # output
    y_output = layers.Dense(n_labels,
                             activation='sigmoid', 
                             kernel_regularizer=L1L2(l1=common_reg, l2=common_reg), 
                             bias_regularizer=L1L2(l1=common_reg, l2=common_reg), 
                             name='output'
                            )(common_layer)
    
    # losses
    bce_loss = K.mean(WCELoss(y_input, y_output, e=1))
    
    # generate model
    all_inputs = [x_input, y_input]
    all_outputs = [y_output]
    
    model = keras.Model(all_inputs, all_outputs)
    
    model.add_loss(bce_loss)
    
    model.compile("adam", [None])
    
    return model



def PhyC_model(params, w_pa=1):
    
    n_inputs = params['n_inputs']
    n_labels = params['n_labels']
    layer_shape = params['layer_shape']
    common_reg = params['common_reg']
    n_train_providers = params['n_train_providers']
    gk_l1l2 = params['gk_l1l2']
    gb_l1l2 = params['gb_l1l2']
    
    
    # inputs
    x_input = keras.Input(shape=(n_inputs,), name='input')
    fk_input = keras.Input(shape=(n_train_providers, n_labels,), name='fk_input')
    y_input = keras.Input(shape=(n_labels,), name='y_input')
    flag_input = keras.Input(shape=(n_train_providers,), name='flag_input')
        
    # g model
    g_multihead_layer = MultiheadDense(units=n_labels,
                                       heads=n_train_providers,
                                       positional_mask=True,
                                       activation='sigmoid',
                                       name='g_multihead_layer',
                                       kernel_regularizer=L1L2(gk_l1l2),
                                       bias_regularizer=L1L2(gb_l1l2))
    
    g_output = g_multihead_layer(fk_input)
    
    # equvalent to min⁡(1,2*o(gk))
    g_output = 2 * (0.5-layers.ReLU()(0.5-g_output))

    # f model
    common_layer = x_input
    
    for i_layer, this_layer_shape in enumerate(layer_shape):
        common_layer = layers.Dense(this_layer_shape, 
                                    activation='relu', 
                                    kernel_regularizer=L1L2(l1=common_reg, l2=common_reg), 
                                    bias_regularizer=L1L2(l1=common_reg, l2=common_reg), 
                                    name='dense_layer_%d' % i_layer
                                   )(common_layer)
    
    f_output = layers.Dense(n_labels,
                            activation='sigmoid', 
                            kernel_regularizer=L1L2(l1=common_reg, l2=common_reg),
                            bias_regularizer=L1L2(l1=common_reg, l2=common_reg),
                            name='common_pred'
                           )(common_layer)

    # losses
    fomula_pred = tf.einsum('ij, ipj->ipj', f_output, g_output)
    anchor_loss = K.mean(K.mean(AnchorLoss(fk_input, fomula_pred), axis=1) * y_input) 
    anchor_loss = w_pa * anchor_loss
    
    e = tf.einsum('npj, np->nj', K.stop_gradient(g_output), flag_input) + epsilon
    wce_loss = K.mean(WCELoss(y_input, f_output, e))
    
    # set inputs and outputs
    all_inputs = [x_input, fk_input, y_input, flag_input]
    all_outputs = [f_output, g_output]

    # generate model
    model = keras.Model(all_inputs, all_outputs)

    model.add_loss(anchor_loss)
    model.add_loss(wce_loss)

    model.compile("adam", [None, None])

    return model