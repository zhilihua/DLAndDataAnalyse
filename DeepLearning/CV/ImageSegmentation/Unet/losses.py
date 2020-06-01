from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

def dice_coeff(y_true, y_pred):
    smooth = 1e-15
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)

    score = (intersection + smooth) / (union - intersection + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = - K.log(dice_coeff(y_true, y_pred))
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss