
from tensorflow.keras.losses import Loss

class TreeSupLoss(Loss):

    def __init__(self, loss_function, tree_loss_weight):
        self.loss_fn = loss_function
        self.weight = tree_loss_weight
        self.name = 'treesuploss'


    def call(self, y_true, net_y_pred, nbdt_y_pred):
        network_loss = self.loss_fn(y_true.numpy().reshape(1, y_true.shape[1]), net_y_pred)
        nbdt_loss = self.loss_fn(y_true, nbdt_y_pred.numpy().reshape(1, nbdt_y_pred.shape[0]))
        return nbdt_loss, network_loss, (self.weight * nbdt_loss) + network_loss
