import tensorflow as tf

class AlphaZeroModel(tf.keras.Model):
    """Neural network model of alphazero.
    
    The model architecture consists as following:
    * 1 convolutional block
    * n residual blocks
    * policy head
    * value head
    
    The final output of residual blocks passes into two seperate heads.
    
    All the `Conv2D` and `Dense` that compose `AlphaZeroModel` 
    have L2 regularizer. The final loss of the model is its 
    `compiled loss + L2 regularization loss`. 
    
    Args:
        n_res (int, optional): Number of residual blocks to be made. 
            Setted to 39 by default.
        n_filter (int, optional): Number of filters in convolutional layers.
            Setted to 256 by default.
        p_head_out_dim (int, optional): Dimension of policy head's 
            output vector. Setted to 19 x 19 = 361 by default.
        v_head_out_dim (int, optional): Dimension of value head's 
            output vector. Setted to 1 by default.
    
    Keyword Args:
        input_shape (tuple): Input shape of the model.
    """
    
    #instance variables 
    _conv_block: 'ConvBlock'
    _res_blocks: list
    _p_head: 'PolicyHead'
    _v_head: 'ValueHead'
    
    def __init__(self, n_res: int=39, n_filter: int=256, 
                 p_head_out_dim: int=361, v_head_out_dim: int=1, **kwargs):
        super(AlphaZeroModel, self).__init__()
        
        if 'input_shape' in kwargs:
            self._conv_block = ConvBlock(n_filter, (3, 3), False, 
                                         input_shape=kwargs['input_shape'])
        else:
            self._conv_block = ConvBlock(n_filter, (3, 3), False)
            
        self._res_blocks = [ResBlock(n_filter) for _ in range(n_res)]
        self._p_head = PolicyHead(p_head_out_dim)
        self._v_head = ValueHead(v_head_out_dim)


    def call(self, inputs, training: bool) -> list:
        conv_block_out = self._conv_block(inputs, training)
        res_block_out = self._res_blocks[0](conv_block_out, training)
        
        for i in range(1, len(self._res_blocks)):
            res_block_out = self._res_blocks[i](res_block_out)
        
        p = self._p_head(res_block_out, training)
        v = self._v_head(res_block_out, training)
        
        return [p, v]
        
   
class ConvBlock(tf.keras.layers.Layer):
    """Convolutional block that composes alphazero model.
    Convolutional block that composes first layer of alphazero model 
    has convolution of 256 filters of kernel size 3 x 3 with stride 1 
    and no residual feature.
    
    The convolutional block consists as following:  
    * A convolution of `n_filter` filters of kernel size `kernel_size` 
      with stride 1
    * Batch normalization
    * A residual connection that adds the input of the convolutional block 
      to the output of batch normalization layer (optional)
    * A recitifier nonlinearity
    
    Data format of `ConvBlock` input should follow `channels_first`.
    
    Args:
        n_filter (int, optional): Number of filters of convolutional layer.
        kernel_size (tuple): Kernel size of convolution layer. 
        residual (bool): Whether this instance uses the residual connection.
        
    Keyword Args: 
        input_shape (tuple): Input shape of the convolutional layer. 
            Recommanded to provide `input_shape` if `ConvBlock` 
            is used first layer of model.
    """
    
    #instance vairables
    _conv2d: tf.keras.layers.Conv2D 
    _bn: tf.keras.layers.BatchNormalization
    _residual: bool
    
    def __init__(self, n_filter: int, kernel_size: tuple, 
                 residual: bool, **kwargs):
        super(ConvBlock, self).__init__()
       
        if 'input_shape' in kwargs:
            self._conv2d = tf.keras.layers.Conv2D(n_filter, kernel_size, 
                    padding='same', data_format='channels_first',
                    kernel_regularizer='l2', bias_regularizer='l2', 
                    input_shape=kwargs['input_shape'])
        else:
            self._conv2d = tf.keras.layers.Conv2D(n_filter, kernel_size, 
                    padding='same', data_format='channels_first',
                    kernel_regularizer='l2', bias_regularizer='l2')
                    
        self._bn = tf.keras.layers.BatchNormalization(axis=1, 
                beta_regularizer='l2', gamma_regularizer='l2')
        self._residual = residual


    def call(self, inputs, training: bool) -> tf.Tensor:
        conv2d_out = self._conv2d(inputs)
        bn_out = self._bn(conv2d_out, training)
        
        if self._residual:
            bn_out = tf.keras.layers.add([bn_out, inputs])
            
        out = tf.keras.activations.relu(bn_out)
        
        return out
    
    
class ResBlock(tf.keras.layers.Layer):
    """Residual block that composes alphazero model.
    
    The residual block consists as following:
    * A convolutional block that has `n_filter` filters of kernel size 3 x 3 
      with stride 1 and no residual connection
    * A convolutional block that has `n_filter` filters of kernel size 3 x 3 
      with stride 1 and residual connection
    
    Args: 
        n_filter (int): Number of filters in convolutional layers. 
    """
    
    #instance variables
    _conv_block1: ConvBlock
    _conv_block2: ConvBlock
     
    def __init__(self, n_filter: int):
        super(ResBlock, self).__init__()
        
        self._conv_block1 = ConvBlock(n_filter, (3, 3), False)
        self._conv_block2 = ConvBlock(n_filter, (3, 3), True)
    
    
    def call(self, inputs, training: bool) -> tf.Tensor:
        conv_block1_out = self._conv_block1(inputs, training)
        conv_block2_out = self._conv_block2(conv_block1_out, training)
        
        return conv_block2_out
    
    
class PolicyHead(tf.keras.layers.Layer):
    """Policy head that composes alphazero model.
    
    The policiy head consists as following:
    * A convolutional block that has 2 filters of kernel size 1 x 1
      with stride 1 and no residual connection 
    * A fully connected linear layer that outputs a vector
    
    Args:
        out_dim (int): Dimension of output vector. 
    """
    
    #instance variables
    _conv_block: ConvBlock
    _fc: tf.keras.layers.Dense 
    
    def __init__(self, out_dim: int):
        super(PolicyHead, self).__init__()
        
        self._conv_block = ConvBlock(2, (1, 1), False)    
        self._fc = tf.keras.layers.Dense(out_dim, activation='softmax', 
                                         kernel_regularizer='l2', 
                                         bias_regularizer='l2')


    def call(self, inputs, training: bool) -> tf.Tensor:
        conv_block_out = self._conv_block(inputs, training)
        fc_out = self._fc(tf.reshape(conv_block_out, 
                                     [conv_block_out.shape[0], -1]))
        
        return fc_out
    
    
class ValueHead(tf.keras.layers.Layer):
    """Vlaue head that composes alphazero model.
    
    The value head consists as following:
    * A convolution block that has 1 filter of kernel size 1 x 1 
      with stride 1 and no residual connection.
    * A fully connected linear layer to a hidden layer of size 256
    * A recitifier nonlinearity
    * A fully connected linear layer to vector
    * A tanh nonlinearity outputting vector in the range [-1, 1]
    
    Args:
        out_dim(int): Dimension of ouput vector
    """
    
    #instance variables
    _conv_block: ConvBlock
    _fc1: tf.keras.layers.Dense
    _fc2: tf.keras.layers.Dense
    
    def __init__(self, out_dim: int):
        super(ValueHead, self).__init__()
        
        self._conv_block = ConvBlock(1, (1, 1), False)
        self._fc1 = tf.keras.layers.Dense(256, activation='relu', 
                                          kernel_regularizer='l2', 
                                          bias_regularizer='l2')
        self._fc2 = tf.keras.layers.Dense(out_dim, activation='tanh',
                                          kernel_regularizer='l2',
                                          bias_regularizer='l2')
        
    
    def call(self, inputs, training: bool) -> tf.Tensor:
        conv_block_out = self._conv_block(inputs, training)
        fc1_out = self._fc1(tf.reshape(conv_block_out, 
                                       [conv_block_out.shape[0], -1])) 
        fc2_out = self._fc2(fc1_out)
        
        return fc2_out       


class AlphaZeroLoss(tf.keras.losses.Loss):
    """Loss of alphazero model.
    
    The equation of loss function is \\
    `l = pow(z - v, 2) - transpose(pi) * log(p)`
    
    L2 regularization is applied since all the layers of `AlphaZeroModel` 
    have L2 regularizer.
    """
    
    def call(self, y_true: list, y_pred: list) -> tf.Tensor:
        se = (y_true[1] - y_pred[1]) ** 2
        cee = tf.reshape(
                tf.keras.losses.categorical_crossentropy(y_true[0], y_pred[0]), 
                [y_true[0].shape[0], 1])
    
        return se - cee
    