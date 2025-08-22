from tensorflow import keras as ks

# This file contains 2 functions that wrap the TensorFlow Keras Model APIs to quickly build convolutional neural
# networks with commonly used general architectures.


# build_cnn
#
# This function builds and compiles a convolutional neural network using Keras' Sequential Model API. This API allows
# you to initialize an empty model and then sequentially add layers to it using its 'add' method. It only allows for one
# input and one output layer (which is sufficient for most of the model types I've used). The structure of the function
# assumes that the model will have a number of convolutional layers at the start followed by a block of dense layers,
# with potential dropout layers after each of these layers. All the activation functions in the body of the network are
# assumed to be the same, but the output layer's activation may differ.
#
# Arguments:
#
#   input_size: (= 200)
#       The number of neurons in the network's input layer. This should match the size of the spectra that will be fed
#       to this model.
#
#   num_kernels: (= (16, 16, 16))
#       A tuple containing the number of kernels (or 'filters') that each convolutional layer will have (in order from
#       input layer onwards). The length of this tuple simultaneously defines how many convolutional layers the network
#       will have.
#
#   kernel_size: (= (16, 32, 48))
#       A tuple containing the sizes (receptive fields) of the convolutional kernels in each of the convolutional
#       layers. This should have a length that is at least that of 'num_kernels'.
#
#   conv_drop: (= (0, 0, 0))
#       A tuple containing the dropout probabilities for each of the convolutional layers. This is the probability that
#       the layers' activations are set to 0 during the forward passes in the training procedure. This should have a
#       length that is at least that of 'num_kernels'.
#
#   dense_units: (= (500, 300, 6))
#       A tuple containing the number of units (neurons) that each of the dense layers will have (including the output
#       layer). The length of this tuple simultaneously defines how many dense layers the network will have. The size of
#       the output layer will depend on the type of network.
#
#   dense_drop: (= (0.2, 0.2))
#       A tuple containing the dropout probabilities for each of the dense layers (excluding the output layer). This is
#       the probability that the layers' activations are set to 0 during the forward passes in the training procedure.
#       This should have a length that is at least one less that of 'num_kernels'.
#
#   activations: (= ('relu', 'sigmoid'))
#       A tuple containing the Keras strings for the activation functions to be used in the network. The first entry
#       is the function that will be used for all the layers in the network's body. The second entry is the function to
#       be used in the output layer.
#
#       'relu'    - Rectified Linear Unit    f(x) = max(0, x)
#       'sigmoid' - Sigmoid Function         f(x) = 1/(1 + exp(-x))
#       'tanh'    - Hyperbolic Tangent       f(x) = 2/(1 + exp(-2x)) - 1
#       'softmax' - Softmax Function         f(x[j]) = exp(x[j]) / Σ_i(exp(x[i]))
#
#       Several more can be used. Google brings up very easy articles and cheat sheets that go into detail, and the
#       TensorFlow Keras documentation will give you the string codes.
#
#   loss: (= 'categorical_crossentropy')
#       The Keras string for the loss function to be used during training.
#
#       'categorical_crossentropy' - Categorical Cross-Entropy
#       'mse'                      - Mean Squared Error
#       'mae'                      - Mean Absolute Error
#
#       Several more can be used. Google brings up very easy articles and cheat sheets that go into detail, and the
#       TensorFlow Keras documentation will give you the string codes.
#
#   metrics: (= 'accuracy')
#       A string, or list of strings, for any other performance measures (in addition to the loss itself) to monitor and
#       record during training. Any loss function can also be used as a metric. Other measures, such as 'accuracy' can
#       be used. The calculation of 'accuracy' may differ depending on the output activation function.
#
#   learning_rate: (= 0.001)
#       During the weight updates in the training phase, after the gradient of the loss function has been computed, this
#       parameter controls how large a step is taken in the minimising direction. I have not experimented much with this
#       - it seems most people use the default value of 0.001 and this has worked very well. Too high and the model
#       can't converge because it leaps over minima, too low and it may converge too slowly and is more likely to get
#       stuck in local minima.
#
#   name: (= None)
#       Give your model a name :) (otherwise it will be given the default name 'sequential' by Keras).
#
# Outputs:
#
#   model:
#       The compiled convolutional neural network model, ready to be trained by calling its fit method.
#
#   hyp_par:
#       A dictionary containing the hyper-paramaters of the model. Hyper-parameters are those parameters that are not
#       optimised during training time (neuron weights and biases), but are instead fixed once and for all. E.g. 'the
#       number of neurons in layer x'. The dictionary is created at this point because it's easier to capture this
#       information from the input arguments to the function, rather than from digging into the model object for layer
#       shapes etc.


def build_cnn_multi_input(
    num_inputs=2,
    input_names=[0, 1],
    input_sizes={0: 200, 1: 6},
    inputKernels={0: (16, 16, 16), 1: None},
    inputKernelSize={0: (16, 32, 48), 1: None},
    inputConvDrop={0: None, 1: None},
    inputDense={0: (500,), 1: (15,)},
    inputDenseDrop={0: (0.2,), 1: (0.2, 0.2)},
    inputActivations={0: ("relu", "sigmoid"), 1: ("relu", "sigmoid")},
    num_kernels=None,
    kernel_size=None,
    conv_drop=None,
    dense_units=(500, 300, 6),
    dense_drop=(0.2, 0.2),
    activations=("relu", "sigmoid"),
    loss="categorical_crossentropy",
    metrics="accuracy",
    learning_rate=0.001,
    name=None,
):
    # define two sets of inputs
    #    inputA = Input(shape=(32,))
    #    inputB = Input(shape=(128,))
    # the first branch operates on the first input
    #    x = Dense(8, activation="relu")(inputA)
    #    x = Dense(4, activation="relu")(x)
    #    x = Model(inputs=inputA, outputs=x)
    # the second branch opreates on the second input
    #    y = Dense(64, activation="relu")(inputB)
    #    y = Dense(32, activation="relu")(y)
    #    y = Dense(4, activation="relu")(y)
    #    y = Model(inputs=inputB, outputs=y)
    # combine the output of the two branches
    #    combined = concatenate([x.output, y.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    #    z = Dense(2, activation="relu")(combined)
    #    z = Dense(1, activation="linear")(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    #    model = Model(inputs=[x.input, y.input], outputs=z)

    ##############################  input networks  ###################
    inputModels = {}
    for k in range(num_inputs):
        input_shape = (input_sizes[input_names[k]], 1)
        inputNow = ks.Input(shape=input_shape, name="Input_{}".format(input_names[k]))
        # Define the shape of the input layer - extra dimension because convolutional layers require this input shape when
        # in 'channels_last' (default) mode. The last dimension would usually accommodate the different colour channels in a
        # typical image.
        iNumKernels = inputKernels[input_names[k]]
        iKernelSizes = inputKernelSize[input_names[k]]
        iActivations = inputActivations[input_names[k]]
        iConvDrop = inputConvDrop[input_names[k]]
        iDense = inputDense[input_names[k]]
        iDenseDrop = inputDenseDrop[input_names[k]]
        if iNumKernels is not None:
            for i in range(len(iNumKernels)):
                # Iterate over the number of convolutional layers to be added.

                if i == 0:
                    inputModels[input_names[k]] = ks.layers.Conv1D(
                        filters=iNumKernels[i],
                        kernel_size=(iKernelSizes[i],),
                        padding="same",
                        activation=iActivations[0],
                        name="Conv_{}_{}".format(input_names[k], i),
                    )(inputNow)
                else:
                    inputModels[input_names[k]] = ks.layers.Conv1D(
                        filters=iNumKernels[i],
                        kernel_size=(iKernelSizes[i],),
                        padding="same",
                        activation=iActivations[0],
                        name="Conv_{}_{}".format(input_names[k], i),
                    )(inputModels[input_names[k]])
                # For the first layer, we need to input our data, but subsequent layers can
                # automatically use the previous layer's output shape. Hence for these layers we set 'input_shape' to a
                # NoneType in a list - if this is passed to a layer, it assumes it should use the previous layer's output
                # shape for its input.

                # Add a convolutional layer with the corresponding number of kernels and kernel sizes from the tuples. The
                # layers' activation functions are all set to the first value in the 'activations' tuple. The padding argument,
                # 'same', appends a number of 0s to either side of the layer's input, depending on the size of the kernels, such
                # that the convolution produces an output that has the same dimensions as the input. Convolution would otherwise
                # downsize the input, and we'd lose some of the information at the borders:
                #
                #  [x x x x x x x x x x]  |  [0 0|x x x x x x x x x x|0 0]  Input vector
                #        [w w w w w]      |            [w w w w w]
                #          \  |  /  →     |              \  |  /  →         Kernel slides along input, calculating dot products
                #            \|/          |                \|/
                #      [y y y y - -]      |      [y y y y y y - - - -]      Output (A 'feature map' of the convolutional kernel)
                # A 1D convolutional kernel of size 5 downsizes the input by 2 on each side. Padding 2 0s on each side means
                # that the output is the same size as the unpadded input. There is also an argument called 'stride_length' which
                # controls how big a step the kernel takes across the input. I always leave this as 1 (default).
                if iConvDrop is not None:
                    inputModels[input_names[k]] = ks.layers.Dropout(
                        rate=iConvDrop[i],
                        name="Conv_{}_{}_Drop".format(input_names[k], i),
                    )(inputModels[input_names[k]])
            # Add a dropout layer following each convolutional layer. I usually have these inactive (dropout probabilities
            # of 0). Some sources recommend against using dropout in convolutional layers.
            inputModels[input_names[k]] = ks.layers.Flatten(
                name="Flatten_{}".format(input_names[k])
            )(inputModels[input_names[k]])
        # Add a flattening layer following the convolutional block. This takes the mutliple feature maps (stacked output
        # vectors from each of the previous layer's kernels) and concatenates them into a single vector. This can then be
        # sent as input to the dense layers.
        if iDense is not None:
            for j in range(len(iDense)):
                # Iterate over the number of dense layers to be added (excluding the final output layer).
                if iNumKernels is None and j == 0:
                    inputModels[input_names[k]] = ks.layers.Dense(
                        iDense[j],
                        activation=iActivations[0],
                        name="Dense_{}_{}".format(input_names[k], j),
                    )(inputNow)
                else:
                    inputModels[input_names[k]] = ks.layers.Dense(
                        iDense[j],
                        activation=iActivations[0],
                        name="Dense_{}_{}".format(input_names[k], j),
                    )(inputModels[input_names[k]])
                # Add a dense layer with the corresponding number of units from the the 'dense_units' tuple. The activation
                # functions are all set to the first value in the 'activations' tuple.
                if iDenseDrop is not None:
                    inputModels[input_names[k]] = ks.layers.Dropout(
                        iDenseDrop[j], name="Dense_{}_{}_Drop".format(input_names[k], j)
                    )(inputModels[input_names[k]])
            # Add a dropout layer following each dense layer. Values of 0.2 for all layers seems to work well enough, but
            # people sometimes use values up to 0.5.
            if iNumKernels is None:
                inputModels[input_names[k]] = ks.layers.Flatten(
                    name="Flatten_{}".format(input_names[k])
                )(inputModels[input_names[k]])
        inputModels[input_names[k]] = ks.Model(
            inputs=inputNow, outputs=inputModels[input_names[k]]
        )

    inputCombined = ks.layers.concatenate(
        [inputModels[input_names[i]].output for i in range(num_inputs)]
    )

    ##############################  Combined network  ###################

    # Define the shape of the input layer - extra dimension because convolutional layers require this input shape when
    # in 'channels_last' (default) mode. The last dimension would usually accommodate the different colour channels in a
    # typical image.
    if num_kernels is not None:
        for i in range(len(num_kernels)):
            # Iterate over the number of convolutional layers to be added.

            if i > 0:
                model = ks.layers.Conv1D(
                    filters=num_kernels[i],
                    kernel_size=(kernel_size[i],),
                    padding="same",
                    activation=activations[0],
                    input_shape=input_shape,
                    name="Conv{}".format(i),
                )(inputCombined)
            else:
                model = ks.layers.Conv1D(
                    filters=num_kernels[i],
                    kernel_size=(kernel_size[i],),
                    padding="same",
                    activation=activations[0],
                    input_shape=input_shape,
                    name="Conv{}".format(i),
                )(model)
            # Add a convolutional layer with the corresponding number of kernels and kernel sizes from the tuples. The
            # layers' activation functions are all set to the first value in the 'activations' tuple. The padding argument,
            # 'same', appends a number of 0s to either side of the layer's input, depending on the size of the kernels, such
            # that the convolution produces an output that has the same dimensions as the input. Convolution would otherwise
            # downsize the input, and we'd lose some of the information at the borders:
            #
            #  [x x x x x x x x x x]  |  [0 0|x x x x x x x x x x|0 0]  Input vector
            #        [w w w w w]      |            [w w w w w]
            #          \  |  /  →     |              \  |  /  →         Kernel slides along input, calculating dot products
            #            \|/          |                \|/
            #      [y y y y - -]      |      [y y y y y y - - - -]      Output (A 'feature map' of the convolutional kernel)
            #
            # A 1D convolutional kernel of size 5 downsizes the input by 2 on each side. Padding 2 0s on each side means
            # that the output is the same size as the unpadded input. There is also an argument called 'stride_length' which
            # controls how big a step the kernel takes across the input. I always leave this as 1 (default).

            model = ks.layers.Dropout(rate=conv_drop[i], name="Conv{}_Drop".format(i))(
                model
            )
            # Add a dropout layer following each convolutional layer. I usually have these inactive (dropout probabilities
            # of 0). Some sources recommend against using dropout in convolutional layers.

        model.add(ks.layers.Flatten(name="Flatten"))
    # Add a flattening layer following the convolutional block. This takes the mutliple feature maps (stacked output
    # vectors from each of the previous layer's kernels) and concatenates them into a single vector. This can then be
    # sent as input to the dense layers.

    for j in range(len(dense_units) - 1):
        # Iterate over the number of dense layers to be added (excluding the final output layer).
        if num_kernels is None and j == 0:
            model = ks.layers.Dense(
                dense_units[j], activation=activations[0], name="Dense{}".format(j)
            )(inputCombined)
        else:
            model = ks.layers.Dense(
                dense_units[j], activation=activations[0], name="Dense{}".format(j)
            )(model)
        # Add a dense layer with the corresponding number of units from the the 'dense_units' tuple. The activation
        # functions are all set to the first value in the 'activations' tuple.

        model = ks.layers.Dropout(dense_drop[j], name="Dense{}_Drop".format(j))(model)
        # Add a dropout layer following each dense layer. Values of 0.2 for all layers seems to work well enough, but
        # people sometimes use values up to 0.5.

    model = ks.layers.Dense(
        dense_units[-1], activation=activations[1], name="DenseOut"
    )(model)

    model = ks.Model(
        inputs=[inputModels[input_names[i]].inputs for i in range(num_inputs)],
        outputs=model,
    )

    if isinstance(metrics, str):
        metrics = [metrics]
    elif isinstance(metrics, tuple):
        metrics = list(metrics)
    # The value 'metrics' passed to the model as it's compiled is expected to be a list, even if there's only one
    # metric. These lines make sure this is satisfied.

    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    # Compile the model, specifying the optimizer, learning rate, loss function and any metrics to be monitored during
    # training. I've exclusively used the Adam optimiser for all the networks I've trained since it was widely
    # recommended and has worked very well.
    #
    # Adam is not an acronym, but does derive from 'Adaptive Moment Estimation'. It (effectively) modifies the learning
    # rate during training based on the recent history of the gradient (exponentially weighted moving average), giving
    # the optimisation process a momentum. Infact, 1st and 2nd moments are computed for each parameter in the search
    # space - each of the model's parameters effectively has its own adaptive learning rate. There are some additional
    # arguments associated with this property including 'beta1' = 0.9 and 'beta2' = 0.999. These control how far back in
    # time the optimizer looks when averaging gradients (exponential decay). However, the default values are generally
    # recommended and have worked well.

    model.summary()
    # Print out a summary of the model's layers.

    hyp_par = {
        "learning_rate": learning_rate,
        "number of inputs": num_inputs,
        "input names": input_names,
        "input_sizes": input_sizes,
        "input filters": inputKernels,
        "input filter size": inputKernelSize,
        "input filter drop": inputConvDrop,
        "input dense units": inputDense,
        "input dense drop": inputDenseDrop,
        "input activation": inputActivations,
        "combined filters": num_kernels,
        "combined filter size": kernel_size,
        "combined filter drop": conv_drop,
        "combined dense units": dense_units,
        "combined dense drop": dense_drop,
        "loss": loss,
        "metrics": metrics,
        "activations": activations,
        "optimizer": str(model.optimizer)[36 : str(model.optimizer).index(" ")],
    }
    # Save the model's hyper-parameters into a dictionary.

    return model, hyp_par


def build_cnn(
    input_size=200,
    num_kernels=(16, 16, 16),
    kernel_size=(16, 32, 48),
    conv_drop=(0, 0, 0),
    dense_units=(500, 300, 6),
    dense_drop=(0.2, 0.2),
    activations=("relu", "softmax"),
    loss="categorical_crossentropy",
    metrics="accuracy",
    learning_rate=0.001,
    name=None,
):
    model = ks.Sequential(name=name)
    # Initialize a basic keras sequential model

    input_shape = (input_size, 1)
    # Define the shape of the input layer - extra dimension because convolutional layers require this input shape when
    # in 'channels_last' (default) mode. The last dimension would usually accommodate the different colour channels in a
    # typical image.

    for i in range(len(num_kernels)):
        # Iterate over the number of convolutional layers to be added.

        if i > 0:
            input_shape = [None]
            # For the first layer, we need to explicate the input shape (see above), but subsequent layers can
            # automatically use the previous layer's output shape. Hence for these layers we set 'input_shape' to a
            # NoneType in a list - if this is passed to a layer, it assumes it should use the previous layer's output
            # shape for its input.

        model.add(
            ks.layers.Conv1D(
                filters=num_kernels[i],
                kernel_size=(kernel_size[i],),
                padding="same",
                activation=activations[0],
                input_shape=input_shape,
                name="Conv{}".format(i),
            )
        )
        # Add a convolutional layer with the corresponding number of kernels and kernel sizes from the tuples. The
        # layers' activation functions are all set to the first value in the 'activations' tuple. The padding argument,
        # 'same', appends a number of 0s to either side of the layer's input, depending on the size of the kernels, such
        # that the convolution produces an output that has the same dimensions as the input. Convolution would otherwise
        # downsize the input, and we'd lose some of the information at the borders:
        #
        #  [x x x x x x x x x x]  |  [0 0|x x x x x x x x x x|0 0]  Input vector
        #        [w w w w w]      |            [w w w w w]
        #          \  |  /  →     |              \  |  /  →         Kernel slides along input, calculating dot products
        #            \|/          |                \|/
        #      [y y y y - -]      |      [y y y y y y - - - -]      Output (A 'feature map' of the convolutional kernel)
        #
        # A 1D convolutional kernel of size 5 downsizes the input by 2 on each side. Padding 2 0s on each side means
        # that the output is the same size as the unpadded input. There is also an argument called 'stride_length' which
        # controls how big a step the kernel takes across the input. I always leave this as 1 (default).

        model.add(ks.layers.Dropout(rate=conv_drop[i], name="Conv{}_Drop".format(i)))
        # Add a dropout layer following each convolutional layer. I usually have these inactive (dropout probabilities
        # of 0). Some sources recommend against using dropout in convolutional layers.

    model.add(ks.layers.Flatten(name="Flatten"))
    # Add a flattening layer following the convolutional block. This takes the mutliple feature maps (stacked output
    # vectors from each of the previous layer's kernels) and concatenates them into a single vector. This can then be
    # sent as input to the dense layers.

    for j in range(len(dense_units) - 1):
        # Iterate over the number of dense layers to be added (excluding the final output layer).

        model.add(
            ks.layers.Dense(
                dense_units[j], activation=activations[0], name="Dense{}".format(j)
            )
        )
        # Add a dense layer with the corresponding number of units from the the 'dense_units' tuple. The activation
        # functions are all set to the first value in the 'activations' tuple.

        model.add(ks.layers.Dropout(dense_drop[j], name="Dense{}_Drop".format(j)))
        # Add a dropout layer following each dense layer. Values of 0.2 for all layers seems to work well enough, but
        # people sometimes use values up to 0.5.

    model.add(
        ks.layers.Dense(dense_units[-1], activation=activations[1], name="DenseOut")
    )
    # Add the model's output layer - its number of units is always the last entry in the 'dense_units' tuple. Its
    # activation is the second value in the 'activations' tuple, which will typically be different from the activation
    # used throughout the network's body.

    if isinstance(metrics, str):
        metrics = [metrics]
    elif isinstance(metrics, tuple):
        metrics = list(metrics)
        # The value 'metrics' passed to the model as it's compiled is expected to be a list, even if there's only one
        # metric. These lines make sure this is satisfied.

    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    # Compile the model, specifying the optimizer, learning rate, loss function and any metrics to be monitored during
    # training. I've exclusively used the Adam optimiser for all the networks I've trained since it was widely
    # recommended and has worked very well.
    #
    # Adam is not an acronym, but does derive from 'Adaptive Moment Estimation'. It (effectively) modifies the learning
    # rate during training based on the recent history of the gradient (exponentially weighted moving average), giving
    # the optimisation process a momentum. Infact, 1st and 2nd moments are computed for each parameter in the search
    # space - each of the model's parameters effectively has its own adaptive learning rate. There are some additional
    # arguments associated with this property including 'beta1' = 0.9 and 'beta2' = 0.999. These control how far back in
    # time the optimizer looks when averaging gradients (exponential decay). However, the default values are generally
    # recommended and have worked well.

    model.summary()
    # Print out a summary of the model's layers.

    hyp_par = {
        "learning_rate": learning_rate,
        "input_size": input_size,
        "filters": num_kernels,
        "kernel_size": kernel_size,
        "conv_drop": conv_drop,
        "dense_units": dense_units,
        "dense_drop": dense_drop,
        "loss": loss,
        "metrics": metrics,
        "activations": activations,
        "optimizer": str(model.optimizer)[36 : str(model.optimizer).index(" ")],
    }
    # Save the model's hyper-parameters into a dictionary.

    return model, hyp_par


########################################################################################################################

# build_cnn_multi_out
#
# This function uses Keras' Functional API to build a convolutional neural network with potentially more than one
# output (i.e. branching). This has only been necessary for one-hot position finding networks where I wanted to feed the
# different outputs for each peak to each other in order to dissuade them from guessing each other's peak positions. It
# also allows the use of individual softmax/categorical cross-entropy outputs for each peak (as opposed to sigmoid/
# binary cross-entropy, which can just as easily be applied to a single concatenated output). The output cross-
# connecting didn't work, and training with multiple outputs is more troublesome due to compatibility issues with the
# tf.data API and Keras Models (have to use numpy arrays in RAM). May be of some use regardless.
#
# Arguments:
#
#   Almost identical to 'build_cnn'. Two additional arguments. Outputs are also the same.
#
#   num_outputs: (= 6)
#       The number of separate output layers (should match the number of peak positions the network is supposed to
#       find).
#
#   connect_outputs: (= False)
#       Whether to connect the output layers to each other: Output 1 fed to output 2, output 2 fed to output 3 etc.:
#
#                            6
#                        5  ___
#                    4  ___/ ↑
#                3  ___/ ↑   |
#            2  ___/ ↑   |   |
#        1  ___/ ↑   |   |   |
#       ___/ ↑   |   |   |   |
#      __↑___|___|___|___|___|__
#    /___________________________\
#                  ↑
#       [Convolutional Layers]


def build_cnn_multi_out(
    num_outputs=6,
    connect_outputs=False,
    input_size=200,
    num_kernels=(16, 16, 16),
    kernel_size=(16, 32, 48),
    conv_drop=(0, 0, 0),
    dense_units=(500, 300, 200),
    dense_drop=(0.2, 0.2),
    activations=("relu", "softmax"),
    loss="categorical_crossentropy",
    metrics="accuracy",
    learning_rate=0.001,
    name=None,
):
    input_layer = ks.Input(shape=(input_size, 1), name="Input")
    # Initialize the input layer with a shape of (/input_size/, 1). The extra dimension is required by Keras/TensorFlow
    # implementation of convolutions (usually it would hold the different colour channels of an image).

    layer = ks.layers.Conv1D(
        filters=num_kernels[0],
        kernel_size=(kernel_size[0],),
        padding="same",
        activation=activations[0],
        name="Conv0",
    )(input_layer)
    # Create the first convolutional layer. It is connected to the input layer by passing the input layer to it as an
    # argument. This method of connecting layers allows you to create more complex architectures than the Sequential
    # API.

    layer = ks.layers.Dropout(rate=conv_drop[0], name="Conv0_Drop")(layer)
    # Create the first dropout layer and connect to the first convolutional layer.

    for i in range(1, len(num_kernels)):
        # Iterate over the remaining number of convolutional layers to be added. The first layer is added outside the
        # loop because it is connected to the input layer, which has a unique name. We need this unique variable name
        # for the input in order to compile the model at the end. For all the intervening layers we can keep overriding
        # the variable 'layer'.

        layer = ks.layers.Conv1D(
            filters=num_kernels[i],
            kernel_size=(kernel_size[i],),
            padding="same",
            activation=activations[0],
            name="Conv{}".format(i),
        )(layer)
        layer = ks.layers.Dropout(rate=conv_drop[i], name="Conv{}_Drop".format(i))(
            layer
        )
        # Add the remaining convolutional and associated dropout layers to the model, joining them in sequence as
        # before.

    layer = ks.layers.Flatten(name="Flatten")(layer)
    # Add the flattening layer, again in sequence.

    for j in range(len(dense_units) - 1):
        # Iterate over the number of dense layers to be added (excluding the final output layer).

        layer = ks.layers.Dense(units=dense_units[j], name="Dense{}".format(j))(layer)
        layer = ks.layers.Dropout(rate=dense_drop[j], name="Dense{}_Drop".format(j))(
            layer
        )
        # Add the dense and associated dropout layers as before.

    if connect_outputs:
        # Connect the separate outputs as shown in the diagram above.

        out_list = [
            ks.layers.Dense(
                units=dense_units[-1], activation=activations[1], name="Out0"
            )(layer)
        ]
        # Initialize a list that will contain the output layer objects, already containing the 0th output.

        concat_list = []
        # Initialize a list that will contain concatenations of the final dense layer (before outputs) and each of the
        # first 5 output layers. The kth output will be connected to the (k-1)th concatenation layer (except for the 0th
        # output which is simply connected to the final dense layer alone.

        for k in range(1, num_outputs):
            # Iterate over the output layers (excluding the 0th output).

            concat_list.append(
                ks.layers.concatenate(
                    [layer, out_list[k - 1]], name="Concat{}".format(k)
                )
            )
            # Concatenate the final dense layer ('layer') with the (k-1)th output and apppend the resulting layer to
            # 'concat_list'.

            out_list.append(
                ks.layers.Dense(
                    units=dense_units[-1],
                    activation=activations[1],
                    name="Out{}".format(k),
                )(concat_list[k - 1])
            )
            # Create the kth output layer, connect it to the (k-1)th concatenation layer and append to 'out_list'.

    else:
        # Don't connect the outputs to each other.

        out_list = []
        # Initialize a list that will contain the output layers.

        for k in range(num_outputs):
            out_list.append(
                ks.layers.Dense(
                    units=dense_units[-1],
                    activation=activations[1],
                    name="Out{}".format(k),
                )(layer)
            )
            # Create /num_outputs/ parallel output layers and append them to 'out_list'.

    model = ks.Model(inputs=input_layer, outputs=out_list, name=name)
    # Create the model by calling the Keras Model inject class on the input layer and the list of output layers.

    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[metrics],
    )
    # Compile the model as before.

    model.summary()
    # Print out a summary of the model's layers.

    hyp_par = {
        "learning_rate": learning_rate,
        "input_size": input_size,
        "filters": num_kernels,
        "kernel_size": kernel_size,
        "conv_drop": conv_drop,
        "dense_units": dense_units,
        "dense_drop": dense_drop,
        "loss": loss,
        "metrics": metrics,
        "activations": activations,
        "optimizer": str(model.optimizer)[36 : str(model.optimizer).index(" ")],
    }
    # Save the model's hyper-parameters into a dictionary.

    return model, hyp_par


def f2(
    self, q, X, Y, xs, rs, rmn, desgm, gaus=None, C1=1, C2=1, C3=0, plotter=True
):  # Function defining the difference between the theoretical and measured yield curves (Chi^2 can be potentially introduced here)
    ys = rs * C1 + 2 * q[1] * C2 * np.sqrt(rs) * np.cos(
        C3 + np.arctan2(np.imag(xs), np.real(xs)) - 2 * np.pi * q[2]
    )
    if gaus is None:
        ysg = ys
    else:
        ysg = fftconvolve(
            gaus, ys, "full"
        )  # Convolution of yield with a Gaussian function
    #
    ysgm = fftconvolve(rmn, ysg) + 1
    # Convolution of the Gaussian convoluted yield with mono rocking curve
    #
    tck = interpolate.splrep(desgm, ysgm, s=0)
    ysgmx = interpolate.splev(X, tck, der=0)
    e = q[0] * Y - ysgmx  # /(dataerr[:,1]**2);
    # print("e shape = " + str(e.shape))
    # plt.plot(X,ysgmx)
    # plt.plot(X,q[0]*Y)
    if plotter:
        return e, ysgm, ysgmx
    else:
        return e

##TODO documentation
def build_2D_cnn(
        input_shape = (504, 200, 1),
        num_conv_layers = 6,
        filters = [16, 32, 48, 64, 80, 96],
        kernels = [5, 5, 5, 5, 5, 5],
        pooling_sizes = [2, 2, 2, 2, 2, 2],
        num_dense_layers = 3,
        dense_neurons = [512, 128, 1],
        dropout_probabilites = [0.2, 0.2],
        learning_rate = 0.001,
        loss = 'mae',
        metrics = ['mse'],
        activations = ['relu', 'sigmoid']
):
    """The equivalent of build_cnn, but for 2D data - the input shape is (504, 200, 1), and the convolutional layers are 2D rather than 1D.
    Adds MaxPooling2D layers after each Conv2D layer to reduce the size (spatial dimensionality) of our data, and therefore the size of our networks"""
    model = ks.Sequential()
    model.add(ks.Input(dtype='float32', shape = input_shape)) #(batch_size, height, width, channels)
    for i in range(num_conv_layers):
        model.add(ks.layers.Conv2D(filters = filters[i], kernel_size = kernels[i], padding = 'same', activation = activations[0]))
        model.add(ks.layers.MaxPooling2D(pool_size = pooling_sizes[i], padding = 'same'))

    model.add(ks.layers.Flatten())

    for i in range(num_dense_layers - 1):
        model.add(ks.layers.Dense(dense_neurons[i], activation=activations[0]))
        model.add(ks.layers.Dropout(dropout_probabilites[i]))
    
    model.add(ks.layers.Dense(dense_neurons[-1], activation = activations[1]))

    model.compile(optimizer = ks.optimizers.Adam(learning_rate = learning_rate  ),
              loss = loss,
              metrics = metrics)
    
    hyper_params = {
        "learning_rate": learning_rate,
        "filters": filters,
        "kernels": kernels,
        "pooling_sizes": pooling_sizes,
        "dropout_probabilites": dropout_probabilites,
        "loss": loss,
        "metrics": metrics
}

    return model, hyper_params

def build_2D_cnn_multi_out(input_shape = (504, 200, 1),
        num_conv_layers = 6,
        filters = [16, 32, 48, 64, 80, 96],
        kernels = [5, 5, 5, 5, 5, 5],
        pooling_sizes = [2, 2, 2, 2, 2, 2],
        num_dense_layers = 3,
        dense_neurons = [512, 128, 1],
        dropout_probabilites = [0.2, 0.2],
        learning_rate = 0.001,
        loss = 'mae',
        metrics = ['mse'],
        activations = ['relu', 'sigmoid']
        ):
    """The equivalent of build_cnn_multi_out, but for 2D data - the input shape is (504, 200, 1), and the convolutional layers are 2D rather than 1D.
    Adds MaxPooling2D layers after each Conv2D layer to reduce the size (spatial dimensionality) of our data, and therefore the size of our networks
    Built using the keras functional API, rather than the sequential used elsewhere."""
    inputs = ks.layers.Input(shape = input_shape)
    x = ks.layers.Conv2D(filters = filters[0], kernel_size = kernels[0], padding = 'same', activation = activations[0])(inputs)
    for i in range(num_conv_layers):
        x = ks.layers.Conv2D(filters = filters[i], kernel_size = kernels[i], padding = 'same', activation = activations[0])(x)
        x = ks.layers.MaxPooling2D(pool_size = pooling_sizes[i], padding = 'same')(x)

    x = ks.layers.Flatten()(x)

    for i in range(num_dense_layers - 1):
        x = ks.layers.Dense(dense_neurons[i], activation=activations[0])(x)
        x = ks.layers.Dropout(dropout_probabilites[i])(x)

    outputs = []
    loss_dict, metrics_dict = {}, {}
    for i in range(dense_neurons[-1]):
        layer_name = 'output_{}'.format(i)
        outputs.append(ks.layers.Dense(units =1, activation = activations[i], name=layer_name)(x))
        loss_dict[layer_name] = loss
        metrics_dict[layer_name] = metrics

    print(loss_dict)
    print(metrics_dict)

    model = ks.Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer = ks.optimizers.Adam(learning_rate = learning_rate),
              loss = loss_dict,
              metrics = metrics_dict)
    
    hyper_params = {
        "learning_rate": learning_rate,
        "filters": filters,
        "kernels": kernels,
        "pooling_sizes": pooling_sizes,
        "dropout_probabilites": dropout_probabilites,
        }
    
    return model, hyper_params