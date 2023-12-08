import tensorflow as tf

class AlphaZeroModel(tf.keras.Model):
    """Neural network model of alphazero.
    
    The model architecture consists as following:
    * 1 convolutional block
    * n residual blocks
    * policy head
    * value head
    
    The final output of residual blocks passes into two seperate heads.
    
    Args:
        n_res (int): Number of residual blocks to be made.
        n_filter (int, optional): Number of filters in convolutional layers.
            Setted to 256 by default.
        p_head_output_dim (int, optional): Dimension of policy head's 
            output vector. Setted to 19 x 19 = 361 by default.
        v_head_output_dim (int, optional): Dimension of value head's 
            output vector. Setted to 1 by default.
    """
    
    #instance variables 
    _conv_block: 'ConvBlock'
    _res_blocks: list
    _p_head: 'PolicyHead'
    _v_head: 'ValueHead'
    
    def __init__(self, n_res: int, n_filter: int=256, 
                 p_head_output_dim: int=361, v_head_ouput_dim: int=1):
        pass


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        pass
        
   
class ConvBlock(tf.keras.layers.Layer):
    """Convolutional block that composes alphazero model.
    
    The convolutional block consists as following:  
    - A convolution of `n_filter` filters of kernel size 3 x 3 with stride 1
    - Batch normalization
    - A recitifier nonlinearity
    
    Args:
        n_filter (int, optional): Number of filters in convolutional layer.
            The value is setted to 256 by default.
        data_format (str, optional) The ordering of dimensions in inputs.
            One of the `channels_last` or `channels_first`. 
            Defaults to `channels_last`.
    Keyword Args:
        input_shape (tuple): Input shape of the convolutional layer. 
            Recommanded to pass the argument if this instance 
            is first layer of model. 
    """
    
    #instance vairables
    _conv2: tf.keras.layers.Conv2D 
    _bn: tf.keras.layers.BatchNormalization

    def __init__(self, n_filter: int=256, data_format: str=None, **kwargs):
        super(ConvBlock, self).__init__()
       
        if "input_shape" in kwargs:
            self._conv2 = tf.keras.layers.Conv2D(n_filter, (3, 3), 
                    data_format=data_format, 
                    input_shape=kwargs['input_shape'])
        else:
            self._conv2 = tf.keras.layers.Conv2D(n_filter, (3, 3),
                    data_format=data_format)
        
        self._bn = tf.keras.layers.BatchNormalization()
    
        
    def call(self, inputs, training) -> tf.Tensor:
        conv2_out = self._conv2(inputs)
        bn_out = self._bn(conv2_out, training)
        out = tf.keras.activations.relu(bn_out)
        
        return out
    
    
class ResBlock(tf.keras.layers.Layer):
    """Residual block that composes alphazero model.
    
    The residual block consists as following:
    * A convolution of 256 filters of kernel size 3 x 3 with stride 1
    * Batch normalization
    * A recitifier nonlinearity
    * A convolution of 256 filters of kernel size 3 x 3 with stride 1
    * Batch normalization
    * A skip connection that adds the input to the block
    * A recitifier nonlinearity
    
    Args: 
        n_filter (int, optional): Number of filters in convolutional layers. 
    """
    
    #instance variables
    _conv2a: tf.keras.layers.Conv2D
    _bn2a: tf.keras.layers.BatchNormalization
    _conv2b: tf.keras.layers.Conv2D
    _bn2b: tf.keras.layers.BatchNormalization
    
    def __init__(self, n_filter: int):
        pass
    
    
    def call(self, inputs: tf.Tensor):
        pass
    
    
class PolicyHead(tf.keras.layers.Layer):
    """Policy head that composes alphazero model.
    
    The policiy head consists as following:
    
    * A convolution of 2 filters of kernel size 1 x 1 with stride 1
    * Batch normalization
    * A recitifier nonlinearity
    * A fully connected linear layer that outputs a vector
    
    Args:
        output_dim (int, optional): Dimension of output vector. 
    """
    
    #instance variables
    _conv2: tf.keras.layers.Conv2D
    _bn2: tf.keras.layers.BatchNormalization
    _fc: tf.keras.layers.Dense 
    
    def __init__(self, output_dim: int):
        pass
    
    
    def call(self, inputs: tf.Tensor):
        pass
    
    
class ValueHead(tf.keras.layers.Layer):
    """Vlaue head that composes alphazero model.
    
    The value head consists as following:
    * A convolution of 1 filter of kernel size 1 x 1 with stride 1
    * Batch normalization
    * A recitifier nonlinearity
    * A fully connected linear layer to a hidden layer of size 256
    * A recitifier nonlinearity
    * A fully connected linear lyaer to a scalr
    * A tanh nonlinearity outputting a scalar in the range [-1, 1]
    """
    
    #instance variables
    _conv2: tf.keras.layers.Conv2D
    _bn2: tf.keras.layers.BatchNormalization
    _fca: tf.keras.layers.Dense
    _fcb: tf.keras.layers.Dense
    
    def __init__(self):
        pass
        
    
    def call(self, inputs: tf.Tensor):
        pass


class AlphaZeroLoss(tf.keras.losses.Loss):
    def __init__(self, reduction: tf.keras.losses.Reduction, name:str):
        pass
    
    
    def call(self, y_true, y_pred):
        pass
