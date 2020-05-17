from tensorflow.keras.layers import Conv2D, \
    MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Model, regularizers

def mymodel(inputs, nums_class=5):
    # 第一层
    x = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    # 第二层
    x = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    # 第三层
    x = Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    # 第四层
    x = Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    # 第五层
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu',
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = Flatten()(x)  # 拉直 7*7*512
    # 第九层
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    # 第十层
    x = Dense(128, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    # 第十一层
    x = Dense(nums_class, activation='softmax')(x)

    model = Model(inputs, x)
    return model