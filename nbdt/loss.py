
from tensorflow.keras.losses import Loss


class TreeSupLoss(Loss):

    def __init__(self, loss_function, tree_loss_weight):
        self.loss_fn = loss_function
        self.weight = tree_loss_weight


    def call(self, y_true, net_y_pred, nbdt_y_pred):
        network_loss = self.loss_fn(y_true, net_y_pred).numpy()
        nbdt_loss = self.loss_fn(y_true, nbdt_y_pred).numpy()
        return network_loss + self.weight * nbdt_loss
