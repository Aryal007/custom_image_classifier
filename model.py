import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


# Define our network architecture:

def setup_model(checkpoint_path=None):
    img_prep = ImagePreprocessing()   
    '''
    (optional step), you can feed data_preprocessing = None and the network will still train
    Data preprocessing is needed to resolve issues such as incompleteness, inconsistency,
    and/or lacking in certain behaviours or trends which occur often in real world data
    '''              
    img_prep.add_featurewise_zero_center()
    """ 
    add_samplewise_zero_center.
    Zero center every sample with specified mean. If not specified,
    the mean is evaluated over all samples.
    Arguments:
        mean: `float` (optional). Provides a custom mean. If none
            provided, it will be automatically caluclated based on
            the training dataset. Default: None.
        per_channel: `bool`. If True, compute mean per color channel.
    Returns:
        Nothing.
    """
    img_prep.add_featurewise_stdnorm()
    """
    add_featurewise_stdnorm.
    Scale each sample by the specified standard deviation. If no std
    specified, std is evaluated over all samples data.
    Arguments:
        std: `float` (optional). Provides a custom standard derivation.
            If none provided, it will be automatically caluclated based on
            the training dataset. Default: None.
        per_channel: `bool`. If True, compute std per color channel.
    Returns:
        Nothing.
    """

    # Create extra synthetic training data by flipping, rotating and blurring the
    # images on our data set.
    img_aug = ImageAugmentation()
    '''
    (optional step), you can feed data_augmentation = None and the network will still train
    Data augmentation increases our data set by performing various operations on images such
    as flipping, rotation and blurring. Also, the training becomes better as the network will 
    be later able to identify blurred images, flipped images, rotated images as well.
    '''    
    img_aug.add_random_flip_leftright()
    """ 
    Randomly flip an image (left to right).
    Returns:
        Nothing.
    """
    img_aug.add_random_rotation(max_angle=25.)
    """ 
    Randomly rotate an image by a random angle (-max_angle, max_angle).
    Arguments:
        max_angle: `float`. The maximum rotation angle.
    Returns:
        Nothing.
    """
    img_aug.add_random_blur(sigma_max=3.)
    """ 
    Randomly blur an image by applying a gaussian filter with a random
    sigma (0., sigma_max).
    Arguments:
        sigma: `float` or list of `float`. Standard deviation for Gaussian
            kernel. The standard deviations of the Gaussian filter are
            given for each axis as a sequence, or as a single number,
            in which case it is equal for all axes.
    Returns:
        Nothing.
    """

    network = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    # Input is a 32x32 image with 3 color channels (red, green and blue)
    # The first placeholder in shape array must be None to denote the batch size
    # If not provided, will be added automatically
    # print network = Tensor("InputData/X:0", shape=(?, 32, 32, 3), dtype=float32)
    """ 
    This layer is used for inputting (aka. feeding) data to a network.
    A TensorFlow placeholder will be used if it is supplied,
    otherwise a new placeholder will be created with the given shape.
    Either a shape or placeholder must be provided, otherwise an
    exception will be raised.
    Input:
        List of `int` (Shape), to create a new placeholder.
            Or
        `Tensor` (Placeholder), to use an existing placeholder.
    Output:
        Placeholder Tensor with given shape.
    Arguments:
        shape: list of `int`. An array or tuple representing input data shape.
            It is required if no placeholder is provided. First element should
            be 'None' (representing batch size), if not provided, it will be
            added automatically.
        placeholder: A Placeholder to use for feeding this layer (optional).
            If not specified, a placeholder will be automatically created.
            You can retrieve that placeholder through graph key: 'INPUTS',
            or the 'placeholder' attribute of this function's returned tensor.
        dtype: `tf.type`, Placeholder data type (optional). Default: float32.
        data_preprocessing: A `DataPreprocessing` subclass object to manage
            real-time data pre-processing when training and predicting (such
            as zero center data, std normalization...).
        data_augmentation: `DataAugmentation`. A `DataAugmentation` subclass
            object to manage real-time data augmentation while training (
            such as random image crop, random image flip, random sequence
            reverse...).
        name: `str`. A name for this layer (optional).
    """
    network = conv_2d(network, 32, 3, activation='relu')
    # conv_2d(incoming, nb_filter, filter_size, strides=1, padding='same',
    #         activation='linear', bias=True, weights_init='uniform_scaling',
    #         bias_init='zeros', regularizer=None, weight_decay=0.001,
    #         trainable=True, restore=True, reuse=False, scope=None,
    #         name="Conv2D"):
    # Input:
    #     4-D Tensor [batch, height, width, in_channels].
    # Output:
    #     4-D Tensor [batch, new height, new width, nb_filter].
    # print network = Tensor("Conv2D/Relu:0", shape=(?, 32, 32, 32), dtype=float32)
    '''
    Here we're using relu activation function instead of the sigmoid function 
    that we looked at as relu has reduced likleyhood of the vanishing gradient
    that we saw in sigmoid

    This particular function takes the previously created network as input, the 
    number of convolutional filters being 32 with each filter size 3. 
    [http://cs231n.github.io/convolutional-networks/]
    '''
    network = max_pool_2d(network, 2)
    # max_pool_2d(incoming, kernel_size, strides=None, padding='same',
    #             name="MaxPool2D"):
    # Input:
    #     4-D Tensor [batch, height, width, in_channels].
    # Output:
    #     4-D Tensor [batch, pooled height, pooled width, in_channels].
    # print network = Tensor("MaxPool2D/MaxPool:0", shape=(?, 16, 16, 32), dtype=float32)
    '''
    It is common to periodically insert a Pooling layer in-between successive Conv 
    layers in a ConvNet architecture. Its function is to progressively reduce the 
    spatial size of the representation to reduce the amount of parameters and 
    computation in the network, and hence to also control overfitting.
    '''
    network = conv_2d(network, 64, 3, activation='relu')
    #print network = Tensor("Conv2D_1/Relu:0", shape=(?, 16, 16, 64), dtype=float32)
    network = conv_2d(network, 64, 3, activation='relu')
    # print network = Tensor("Conv2D_2/Relu:0", shape=(?, 16, 16, 64), dtype=float32)
    network = max_pool_2d(network, 2)
    # print network = Tensor("MaxPool2D_1/MaxPool:0", shape=(?, 8, 8, 64), dtype=float32)
    network = fully_connected(network, 512, activation='relu')
    # fully_connected(incoming, n_units, activation='linear', bias=True,
    #                 weights_init='truncated_normal', bias_init='zeros',
    #                 regularizer=None, weight_decay=0.001, trainable=True,
    #                 restore=True, reuse=False, scope=None,
    #                 name="FullyConnected"):
    #   incoming: `Tensor`. Incoming (2+)D Tensor.
    #   n_units: `int`, number of units for this layer.
    '''
    Neurons in a fully connected layer have full connections to all activations in the 
    previous layer, as seen in regular Neural Networks. Their activations can hence be
    computed with a matrix multiplication followed by a bias offset. 
    '''
    # print network = Tensor("FullyConnected/Relu:0", shape=(?, 512), dtype=float32)
    network = dropout(network, 0.5)
    # it's one of the method to prevent overfitting
    """ Dropout.
    Outputs the input element scaled up by `1 / keep_prob`. The scaling is so
    that the expected sum is unchanged.
    Arguments:
        incoming : A `Tensor`. The incoming tensor.
        keep_prob : A float representing the probability that each element
            is kept.
        name : A name for this layer (optional).
    References:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
        N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
        (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    Links:
      [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf]
        (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
    """
    # print network = Tensor("Dropout/cond/Merge:0", shape=(?, 512), dtype=float32)
    network = fully_connected(network, 2, activation='softmax')
    # print network = Tensor("FullyConnected_1/Softmax:0", shape=(?, 2), dtype=float32)
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    """
    Regression.
    The regression layer is used in TFLearn to apply a regression (linear or
    logistic) to the provided input. It requires to specify a TensorFlow
    gradient descent optimizer 'optimizer' that will minimize the provided
    loss function 'loss' (which calculate the errors). A metric can also be
    provided, to evaluate the model performance.
    A 'TrainOp' is generated, holding all information about the optimization
    process. It is added to TensorFlow collection 'tf.GraphKeys.TRAIN_OPS'
    and later used by TFLearn 'models' classes to perform the training.
    An optional placeholder 'placeholder' can be specified to use a custom
    TensorFlow target placeholder instead of creating a new one. The target
    placeholder is added to the 'tf.GraphKeys.TARGETS' TensorFlow
    collection, so that it can be retrieved later.
    Additionaly, a list of variables 'trainable_vars' can be specified,
    so that only them will be updated when applying the backpropagation
    algorithm.
    Input:
        2-D Tensor Layer.
    Output:
        2-D Tensor Layer (Same as input).

    Arguments:
        incoming: `Tensor`. Incoming 2-D Tensor.
        placeholder: `Tensor`. This regression target (label) placeholder.
            If 'None' provided, a placeholder will be added automatically.
            You can retrieve that placeholder through graph key: 'TARGETS',
            or the 'placeholder' attribute of this function's returned tensor.
        optimizer: `str` (name), `Optimizer` or `function`. Optimizer to use.
            Default: 'adam' (Adaptive Moment Estimation).
        loss: `str` (name) or `function`. Loss function used by this layer
            optimizer. Default: 'categorical_crossentropy'.
        metric: `str`, `Metric` or `function`. The metric to be used.
            Default: 'default' metric is 'accuracy'. To disable metric
            calculation, set it to 'None'.
        learning_rate: `float`. This layer optimizer's learning rate.
        dtype: `tf.types`. This layer placeholder type. Default: tf.float32.
        batch_size: `int`. Batch size of data to use for training. tflearn
            supports different batch size for every optimizers. Default: 64.
        shuffle_batches: `bool`. Shuffle or not this optimizer batches at
            every epoch. Default: True.
        to_one_hot: `bool`. If True, labels will be encoded to one hot vectors.
            'n_classes' must then be specified.
        n_classes: `int`. The total number of classes. Only required when using
            'to_one_hot' option.
        trainable_vars: list of `Variable`. If specified, this regression will
            only update given variable weights. Else, all trainale variable
            are going to be updated.
        restore: `bool`. If False, variables related to optimizers such
            as moving averages will not be restored when loading a
            pre-trained model.
        op_name: A name for this layer optimizer (optional).
            Default: optimizer op name.
        validation_monitors: `list` of `Tensor` objects.  List of variables
            to compute during validation, which are also used to produce
            summaries for output to TensorBoard.  For example, this can be
            used to periodically record a confusion matrix or AUC metric, 
            during training.  Each variable should have rank 1, i.e. 
            shape [None].
        validation_batch_size: `int` or None. Specifies the batch
            size to be used for the validation data feed.
        name: A name for this layer's placeholder scope.
    Attributes:
        placeholder: `Tensor`. Placeholder for feeding labels.
    """
    if checkpoint_path:
        model = tflearn.DNN(network, tensorboard_verbose=3,
                            checkpoint_path=checkpoint_path)
    else:
        model = tflearn.DNN(network, tensorboard_verbose=3)
    return model