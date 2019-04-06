import torch
import torch.nn as nn

#  Here I will try to implement an Intrinsic Curiosity Model


class FeatureExtractNet(nn.Module):
    """
    Here we extract the features from the pixels to be used in the model
    """
    def __init__(self, input_dim, filters=16):
        super(FeatureExtractNet, self).__init__()
        #  Using a CNN to extract the features to more easily process the observations
        (channels, height, width) = input_dim
        self.feature_dim = filters * 4 * (height / 8) * (width / 8)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=(5, 5),
                      stride=2, padding=(2, 2)),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters * 2, kernel_size=(5, 5),
                      stride=2, padding=(2, 2)),
            nn.BatchNorm2d(filters * 2),
            nn.LeakyReLU(),
            nn.Conv2d(filters * 2, filters * 4, kernel_size=(5, 5),
                      stride=2, padding=(2, 2)),
            nn.BatchNorm2d(filters*4),
            nn.LeakyReLU(),
            Flatten()
        )

    def forward(self, obs, obs_next):
        #  Converting the input to torch tensor on GPU
        obs = torch.tensor(obs).to(self.device)
        return self.feature_extraction(obs)


class ICMNet(nn.Module):
    def __init__(self, action_space, feature_dim, learning_rate=0.001):
        super(ICMNet, self).__init__()

        #  The action inverter uses the last state and the current state and tries to predict the action taken
        self.action_inverter = nn.Sequential(
            nn.Linear(feature_dim*2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_space)
        )

        #  The forward model tries to predict the upcoming state
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim+action_space, 512),
            nn.LeakyReLU(),
            nn.Linear(512, feature_dim)
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(lr=learning_rate)

    def forward(self, pre_features, features, action):
        #  Concatenation of the features and the actions
        cat_feature_action = torch.cat((action, features))
        #  Feeding the concatenated tensor into the guessing network
        pred_state = self.forward_model(cat_feature_action)
        #  Concatenation of the previous features and the current features
        concat_features = torch.cat((features, pre_features))
        #  Prediction of the action taken
        pred_action = self.action_inverter(concat_features)
        return features, pred_state, pred_action


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ICMLoss(nn.Module):
    """
    A specialized loss function to use for the Internal Curiosity Model
    """
    def __init__(self):
        super(ICMLoss, self).__init__()

    def forward(self, action, predicted_action, features,
                predicted_features, rew_coef=0.5, beta_coef=0.5, eta_coef=0.5):
        #  Loss for feature prediction
        loss_features = 0.5*torch.dist(features, predicted_features)
        #  Loss for action prediction
        loss_action = 0.5*torch.dist(predicted_action, action)
        #  The intrinsic reward
        reward = eta_coef*0.5*torch.dist(predicted_action, action)

        return -rew_coef*reward + (1-beta_coef)*loss_action+beta_coef*loss_features
