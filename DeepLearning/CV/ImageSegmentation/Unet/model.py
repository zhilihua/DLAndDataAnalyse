from tensorflow.keras.models import Model
from tensorflow.keras.layers import InputSpec, Layer
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K

class ConvRelu(Layer):
    def __init__(self, out, **kwargs):
        super(ConvRelu, self).__init__(**kwargs)
        self.out = out
        self.conv = Conv2D(filters=out, kernel_size=(3, 3), kernel_initializer="he_normal",
                           padding='same', activation='relu')
        # self.activation = Activation("relu")

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(ConvRelu, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        # x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.out
        return tuple(output_shape)

    def get_config(self):
        config = {
            'out': self.out,
        }
        base_config = super(ConvRelu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DecoderBlock(Layer):
    def __init__(self, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.out_channels = out_channels

        self.ConvRelu = ConvRelu(middle_channels)
        self.conv2dtran = Conv2DTranspose(out_channels, (3, 3), strides=(2, 2),
                                     padding='same', activation='relu')
        # self.activate = Activation("relu")

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecoderBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # self.block = self.activate(self.conv2dtran(self.ConvRelu(inputs)))
        self.block = self.conv2dtran(self.ConvRelu(inputs))
        return self.block

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.out_channels
        return tuple(output_shape)

    def get_config(self):
        config = {
            'out_channels': self.out_channels,
        }
        base_config = super(DecoderBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def UNet16(inputs, weight_path=None, num_classes=1, num_filters=32):
        vgg_model = VGG16(input_tensor=inputs, weights=weight_path, include_top=False)

        layers = dict([(layer.name, layer) for layer in vgg_model.layers])

        #编码过程
        vgg_top = layers['block5_conv3'].output

        block1_conv2 = layers['block1_conv2'].output
        block2_conv2 = layers['block2_conv2'].output
        block3_conv3 = layers['block3_conv3'].output
        block4_conv3 = layers['block4_conv3'].output

        #定义解码过程
        neck = MaxPooling2D(pool_size=2)(vgg_top)

        center = DecoderBlock(num_filters * 8 * 2, num_filters*8)(neck)
        dec5 = DecoderBlock(num_filters * 8 * 2, num_filters * 8)(concatenate([center, vgg_top]))
        dec4 = DecoderBlock(num_filters * 8 * 2, num_filters * 8)(concatenate([dec5, block4_conv3]))
        dec3 = DecoderBlock(num_filters * 4 * 2, num_filters * 2)(concatenate([dec4, block3_conv3]))
        dec2 = DecoderBlock(num_filters * 2 * 2, num_filters)(concatenate([dec3, block2_conv2]))
        dec1 = ConvRelu(num_filters)(concatenate([dec2, block1_conv2]))
        final = Conv2D(num_classes, (1, 1), activation='sigmoid')

        if num_classes > 1:
            x_out = K.log(K.softmax(dec1))
            model = Model(inputs=vgg_model.input, outputs=[x_out])
        else:
            x_out = final(dec1)
            model = Model(inputs=vgg_model.input, outputs=[x_out])

        # model.summary()
        return model