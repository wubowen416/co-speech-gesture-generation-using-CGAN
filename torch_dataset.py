import numpy as np
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class TrainSet(Dataset):

    def __init__(self):

        # Load data
        with open('data/X_train.p', 'rb') as f:
            X = pickle.load(f)
        with open('data/Y_train.p', 'rb') as f:
            Y = pickle.load(f)
        
        X_concat = np.concatenate(X, axis=0)
        Y_concat = np.concatenate(Y, axis=0)

        x_scaler = MinMaxScaler((-1, 1)).fit(X_concat)
        y_scaler = MinMaxScaler((-1, 1)).fit(Y_concat)
        X_scaled = list(map(x_scaler.transform, X))
        Y_scaled = list(map(y_scaler.transform, Y))

        self.X = X
        self.Y = Y
        self.X_ori = X
        self.Y_ori = Y
        self.X_scaled = X_scaled
        self.Y_scaled = Y_scaled
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
    
    def scaling(self, flag):
        if flag:
            self.X = self.X_scaled
            self.Y = self.Y_scaled
        else:
            self.X = self.X_ori
            self.Y = self.Y_ori

    def scale_x(self, x):
        assert len(x.shape) == 2, "shape of y must be (t, dim)"
        return self.x_scaler.transform(x)

    def rescale_y(self, y):
        assert len(y.shape) == 2, "shape of y must be (t, dim)"
        return self.y_scaler.inverse_transform(y)




if __name__ == "__main__":
    
    seq_set = TrainSet()
    seq_set.scaling(True)

    print(seq_set[0][1].shape)



