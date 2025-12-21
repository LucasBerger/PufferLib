from types import SimpleNamespace
from typing import Any, Tuple

from gymnasium import spaces

from torch import nn
import torch
from torch.distributions.normal import Normal
from torch import nn
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy
from pufferlib.models import Convolutional as Conv
Recurrent = pufferlib.models.LSTMWrapper
from pufferlib.pytorch import layer_init, _nativize_dtype, nativize_tensor
import numpy as np


class Boids(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(4, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
        )
        self.action_vec = tuple(env.single_action_space.nvec)
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, sum(self.action_vec)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        batch, n, = observations.shape
        return self.network(observations.reshape(batch, n//4, 4)).max(dim=1)[0]

    def decode_actions(self, flat_hidden, state=None):
        value = self.value_fn(flat_hidden)
        action = self.actor(flat_hidden).split(self.action_vec, dim=1)
        return action, value

class NMMO3LSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size, hidden_size)

class NMMO3(nn.Module):
    def __init__(self, env, hidden_size=512, output_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        #self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.factors = np.array([4, 4, 17, 5, 3, 5, 5, 5, 7, 4])
        offsets = torch.tensor([0] + list(np.cumsum(self.factors)[:-1])).view(1, -1, 1, 1)
        self.register_buffer('offsets', offsets)
        self.cum_facs = np.cumsum(self.factors)

        self.multihot_dim = self.factors.sum()
        self.is_continuous = False

        self.map_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.multihot_dim, 128, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(128, 128, 3, stride=1)),
            nn.Flatten(),
        )

        self.player_discrete_encoder = nn.Sequential(
            nn.Embedding(128, 32),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(1817, hidden_size)),
            nn.ReLU(),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def forward(self, x, state=None):
        hidden = self.encode_observations(x)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        batch = observations.shape[0]
        ob_map = observations[:, :11*15*10].view(batch, 11, 15, 10)
        ob_player = observations[:, 11*15*10:-10]
        ob_reward = observations[:, -10:]

        batch = ob_map.shape[0]
        map_buf = torch.zeros(batch, 59, 11, 15, dtype=torch.float32, device=observations.device)
        codes = ob_map.permute(0, 3, 1, 2) + self.offsets
        map_buf.scatter_(1, codes, 1)
        ob_map = self.map_2d(map_buf)

        player_discrete = self.player_discrete_encoder(ob_player.int())

        obs = torch.cat([ob_map, player_discrete, ob_player.to(ob_map.dtype), ob_reward], dim=1)
        obs = self.proj(obs)
        return obs

    def decode_actions(self, flat_hidden):
        flat_hidden = self.layer_norm(flat_hidden)
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class Terraform(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.local_net_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(2, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.global_net_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(2, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.net_1d = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(5, hidden_size)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size + cnn_channels*5, hidden_size)),
            nn.ReLU(),
        )
        self.atn_dim = env.single_action_space.nvec.tolist()
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        # breakpoint()
        obs_2d = observations[:, :242].reshape(-1, 2, 11, 11).float()
        obs_1d = observations[:, 242:247].reshape(-1, 5).float()
        location_2d = observations[:, 247:].reshape(-1,2, 6, 6).float()
        hidden_local_2d = self.local_net_2d(obs_2d)
        hidden_global_2d = self.global_net_2d(location_2d)
        hidden_1d = self.net_1d(obs_1d)
        hidden = torch.cat([hidden_local_2d, hidden_global_2d, hidden_1d], dim=1)
        return self.proj(hidden)

    def decode_actions(self, hidden):
        action = self.actor(hidden)
        action = torch.split(action, self.atn_dim, dim=1)
        #action = [head(hidden) for head in self.actor]
        value = self.value(hidden)
        return action, value


class Snake(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        encode_dim = cnn_channels

        '''
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(8, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(encode_dim, hidden_size)),
            nn.ReLU(),
        )
 
        '''
        self.encoder= torch.nn.Sequential(
            nn.Linear(8*np.prod(env.single_observation_space.shape), hidden_size),
            nn.GELU(),
        )
        self.decoder = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        #observations = F.one_hot(observations.long(), 8).permute(0, 3, 1, 2).float()
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        observations = F.one_hot(observations.long(), 8).view(-1, 11*11*8).float()
        return self.encoder(observations)

    def decode_actions(self, hidden):
        action = self.decoder(hidden)
        value = self.value(hidden)
        return action, value

'''
class Snake(pufferlib.models.Default):
    def __init__(self, env, hidden_size=128):
        super().__init__()

    def encode_observations(self, observations, state=None):
        observations = F.one_hot(observations.long(), 8).view(-1, 11*11*8).float()
        super().encode_observations(observations, state)
'''

class Grid(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(32, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.Flatten(),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size)),
            nn.ReLU(),
        )

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))
        else:
            num_actions = env.single_action_space.n
            self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_actions), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        hidden = observations.view(-1, 11, 11).long()
        hidden = F.one_hot(hidden, 32).permute(0, 3, 1, 2).float()
        hidden = self.network(hidden)
        return hidden

    def decode_actions(self, flat_hidden, state=None):
        value = self.value_fn(flat_hidden)
        if self.is_continuous:
            mean = self.decoder_mean(flat_hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            batch = flat_hidden.shape[0]
            return probs, value
        else:
            action = self.actor(flat_hidden)
            return action, value

class Go(nn.Module):
    def __init__(self, env, cnn_channels=64, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        # 3 categories 2 boards. 
        # categories = player, opponent, empty
        # boards = current, previous
        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(2, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride = 1)),
            nn.Flatten(),
        )

        obs_size = env.single_observation_space.shape[0]
        self.grid_size = int(np.sqrt((obs_size-2)/2))
        output_size = self.grid_size - 4
        cnn_flat_size = cnn_channels * output_size * output_size
        
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(2,32))
        
        self.proj = pufferlib.pytorch.layer_init(nn.Linear(cnn_flat_size + 32, hidden_size))

        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1), std=1)
   
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        grid_size = int(np.sqrt((observations.shape[1] - 2) / 2))
        full_board = grid_size * grid_size 
        black_board = observations[:, :full_board].view(-1,1, grid_size,grid_size).float()
        white_board = observations[:, full_board:-2].view(-1,1, grid_size, grid_size).float()
        board_features = torch.cat([black_board, white_board],dim=1)
        flat_feature1 = observations[:, -2].unsqueeze(1).float()
        flat_feature2 = observations[:, -1].unsqueeze(1).float()
        # Pass board through cnn
        cnn_features = self.cnn(board_features)
        # Pass extra feature
        flat_features = torch.cat([flat_feature1, flat_feature2],dim=1)
        flat_features = self.flat(flat_features)
        # pass all features
        features = torch.cat([cnn_features, flat_features], dim=1)
        features = F.relu(self.proj(features))

        return features

    def decode_actions(self, flat_hidden, state=None):
        value = self.value_fn(flat_hidden)
        action = self.actor(flat_hidden)
        return action, value
    
class MOBA(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(16 + 3, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.Flatten(),
        )
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(26, 128))
        self.proj = pufferlib.pytorch.layer_init(nn.Linear(128+cnn_channels, hidden_size))

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()
            self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        cnn_features = observations[:, :-26].view(-1, 11, 11, 4).long()
        map_features = F.one_hot(cnn_features[:, :, :, 0], 16).permute(0, 3, 1, 2).float()
        extra_map_features = (cnn_features[:, :, :, -3:].float() / 255).permute(0, 3, 1, 2)
        cnn_features = torch.cat([map_features, extra_map_features], dim=1)
        #print('observations 2d: ', map_features[0].cpu().numpy().tolist())
        cnn_features = self.cnn(cnn_features)
        #print('cnn features: ', cnn_features[0].detach().cpu().numpy().tolist())

        flat_features = observations[:, -26:].float() / 255.0
        #print('observations 1d: ', flat_features[0, 0])
        flat_features = self.flat(flat_features)
        #print('flat features: ', flat_features[0].detach().cpu().numpy().tolist())

        features = torch.cat([cnn_features, flat_features], dim=1)
        features = F.relu(self.proj(F.relu(features)))
        #print('features: ', features[0].detach().cpu().numpy().tolist())
        return features

    def decode_actions(self, flat_hidden):
        #print('lstm: ', flat_hidden[0].detach().cpu().numpy().tolist())
        value = self.value_fn(flat_hidden)
        if self.is_continuous:
            mean = self.decoder_mean(flat_hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            batch = flat_hidden.shape[0]
            return probs, value
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

            #argmax_samples = [torch.argmax(a, dim=1).detach().cpu().numpy().tolist() for a in action]
            #print('argmax samples: ', argmax_samples)

            return action, value

class TrashPickup(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(5, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size)),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        observations = observations.view(-1, 5, 11, 11).float()
        return self.network(observations)

    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class TowerClimbLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size = 256, hidden_size = 256):
        super().__init__(env, policy, input_size, hidden_size)

class TowerClimb(nn.Module):
    def __init__(self, env, cnn_channels=16, hidden_size = 256, **kwargs):
        self.hidden_size = hidden_size
        self.is_continuous = False
        super().__init__()
        self.network = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Conv3d(1, cnn_channels, 3, stride = 1)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(
                    nn.Conv3d(cnn_channels, cnn_channels, 3, stride=1)),
                nn.Flatten()       
        )
        cnn_flat_size = cnn_channels * 1 * 1 * 5

        # Process player obs
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(3,16))

        # combine
        self.proj = pufferlib.pytorch.layer_init(
                nn.Linear(cnn_flat_size + 16, hidden_size))
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std = 0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1 ), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        board_state = observations[:,:225]
        player_info = observations[:, -3:] 
        board_features = board_state.view(-1, 1, 5,5,9).float()
        cnn_features = self.network(board_features)
        flat_features = self.flat(player_info.float())
        
        features = torch.cat([cnn_features,flat_features],dim = 1)
        features = self.proj(features)
        return features
    
    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        
        return action, value


class ImpulseWarsLSTM(Recurrent):
    def __init__(self, env: pufferlib.PufferEnv, policy: nn.Module, input_size: int = 512, hidden_size: int = 512):
        super().__init__(env, policy, input_size, hidden_size)


class ImpulseWarsPolicy(nn.Module):
    def __init__(
        self,
        env: pufferlib.PufferEnv,
        cnn_channels: int = 64,
        weapon_type_embedding_dims: int = 2,
        input_size: int = 512,
        hidden_size: int = 512,
        batch_size: int = 131_072,
        num_drones: int = 2,
        continuous: bool = False,
        is_training: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.is_continuous = continuous

        self.numDrones = num_drones
        self.isTraining = is_training
        from pufferlib.ocean.impulse_wars import binding
        self.obsInfo = SimpleNamespace(**binding.get_consts(self.numDrones))

        self.discreteFactors = np.array(
            [self.obsInfo.wallTypes] * self.obsInfo.numNearWallObs
            + [self.obsInfo.wallTypes + 1] * self.obsInfo.numFloatingWallObs
            + [self.numDrones + 1] * self.obsInfo.numProjectileObs,
        )
        discreteOffsets = torch.tensor([0] + list(np.cumsum(self.discreteFactors)[:-1])).view(
            1, -1
        )
        self.register_buffer("discreteOffsets", discreteOffsets, persistent=False)
        self.discreteMultihotDim = self.discreteFactors.sum()

        multihotBuffer = torch.zeros(batch_size, self.discreteMultihotDim)
        self.register_buffer("multihotOutput", multihotBuffer, persistent=False)

        # most of the observation is a 2D array of bytes, but the end
        # contains around 200 floats; this allows us to treat the end
        # of the observation as a float array
        _, *self.dtype = _nativize_dtype(
            np.dtype((np.uint8, (self.obsInfo.continuousObsBytes,))),
            np.dtype((np.float32, (self.obsInfo.continuousObsSize,))),
        )
        self.dtype = tuple(self.dtype)

        self.weaponTypeEmbedding = nn.Embedding(self.obsInfo.weaponTypes, weapon_type_embedding_dims)

        # each byte in the map observation contains 4 values:
        # - 2 bits for wall type
        # - 1 bit for is floating wall
        # - 1 bit for is weapon pickup
        # - 3 bits for drone index
        self.register_buffer(
            "unpackMask",
            torch.tensor([0x60, 0x10, 0x08, 0x07], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer("unpackShift", torch.tensor([5, 4, 3, 0], dtype=torch.uint8), persistent=False)

        self.mapObsInputChannels = (self.obsInfo.wallTypes + 1) + 1 + 1 + self.numDrones
        self.mapCNN = nn.Sequential(
            layer_init(
                nn.Conv2d(
                    self.mapObsInputChannels,
                    cnn_channels,
                    kernel_size=5,
                    stride=3,
                )
            ),
            nn.ReLU(),
            layer_init(nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnnOutputSize = self._computeCNNShape()

        featuresSize = (
            cnnOutputSize
            + (self.obsInfo.numNearWallObs * (self.obsInfo.wallTypes + self.obsInfo.nearWallPosObsSize))
            + (
                self.obsInfo.numFloatingWallObs
                * (self.obsInfo.wallTypes + 1 + self.obsInfo.floatingWallInfoObsSize)
            )
            + (
                self.obsInfo.numWeaponPickupObs
                * (weapon_type_embedding_dims + self.obsInfo.weaponPickupPosObsSize)
            )
            + (
                self.obsInfo.numProjectileObs
                * (weapon_type_embedding_dims + self.obsInfo.projectileInfoObsSize + self.numDrones + 1)
            )
            + ((self.numDrones - 1) * (weapon_type_embedding_dims + self.obsInfo.enemyDroneObsSize))
            + (self.obsInfo.droneObsSize + weapon_type_embedding_dims)
            + self.obsInfo.miscObsSize
        )

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(featuresSize, input_size)),
            nn.ReLU(),
        )

        if self.is_continuous:
            self.actorMean = layer_init(nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.actorLogStd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))
        else:
            self.actionDim = env.single_action_space.nvec.tolist()
            self.actor = layer_init(nn.Linear(hidden_size, sum(self.actionDim)), std=0.01)

        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, obs: torch.Tensor, state = None) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_observations(obs)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def unpack(self, batchSize: int, obs: torch.Tensor) -> torch.Tensor:
        # prepare map obs to be unpacked
        mapObs = obs[:, : self.obsInfo.mapObsSize].reshape((batchSize, -1, 1))
        # unpack wall types, weapon pickup types, and drone indexes
        mapObs = (mapObs & self.unpackMask) >> self.unpackShift
        # reshape so channels are first, required for torch conv2d
        return mapObs.permute(0, 2, 1).reshape(
            (batchSize, 4, self.obsInfo.mapObsRows, self.obsInfo.mapObsColumns)
        )

    def encode_observations(self, obs: torch.Tensor, state: Any = None) -> torch.Tensor:
        batchSize = obs.shape[0]

        mapObs = self.unpack(batchSize, obs)

        # one hot encode wall types
        wallTypeObs = mapObs[:, 0, :, :].long()
        wallTypes = F.one_hot(wallTypeObs, self.obsInfo.wallTypes + 1).permute(0, 3, 1, 2).float()

        # unsqueeze floating wall booleans (is wall a floating wall)
        floatingWallObs = mapObs[:, 1, :, :].unsqueeze(1)

        # unsqueeze map pickup booleans (does map tile contain a weapon pickup)
        mapPickupObs = mapObs[:, 2, :, :].unsqueeze(1)

        # one hot drone indexes
        droneIndexObs = mapObs[:, 3, :, :].long()
        droneIndexes = F.one_hot(droneIndexObs, self.numDrones).permute(0, 3, 1, 2).float()

        # combine all map observations and feed through CNN
        mapObs = torch.cat((wallTypes, floatingWallObs, mapPickupObs, droneIndexes), dim=1)
        map = self.mapCNN(mapObs)

        # process discrete observations
        multihotInput = (
            obs[:, self.obsInfo.nearWallTypesObsOffset : self.obsInfo.projectileTypesObsOffset]
            + self.discreteOffsets
        )
        multihotOutput = self.multihotOutput[:batchSize].zero_()
        multihotOutput.scatter_(1, multihotInput.long(), 1)

        weaponTypeObs = obs[:, self.obsInfo.projectileTypesObsOffset : self.obsInfo.discreteObsSize].int()
        weaponTypes = self.weaponTypeEmbedding(weaponTypeObs).float()
        weaponTypes = torch.flatten(weaponTypes, start_dim=1, end_dim=-1)

        # process continuous observations
        continuousObs = nativize_tensor(obs[:, self.obsInfo.continuousObsOffset :], self.dtype)
        # combine all observations and feed through final linear encoder
        features = torch.cat((map, multihotOutput, weaponTypes, continuousObs), dim=-1)

        return self.encoder(features)

    def decode_actions(self, hidden: torch.Tensor):
        if self.is_continuous:
            actionMean = self.actorMean(hidden)
            if self.isTraining:
                actionLogStd = self.actorLogStd.expand_as(actionMean)
                actionStd = torch.exp(actionLogStd)
                action = Normal(actionMean, actionStd)
            else:
                action = actionMean
        else:
            action = self.actor(hidden)
            action = torch.split(action, self.actionDim, dim=1)

        value = self.critic(hidden)

        return action, value

    def _computeCNNShape(self) -> int:
        mapSpace = spaces.Box(
            low=0,
            high=1,
            shape=(self.mapObsInputChannels, self.obsInfo.mapObsRows, self.obsInfo.mapObsColumns),
            dtype=np.float32,
        )

        with torch.no_grad():
            t = torch.as_tensor(mapSpace.sample()[None])
            return self.mapCNN(t).shape[1]

class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(7, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size))
        )
        max_road_objects = 13
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(max_road_objects, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size))
        )
        max_partner_objects = 7
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(max_partner_objects, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size))
        )


        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3*input_size,  hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        self.atn_dim = env.single_action_space.nvec.tolist()
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, sum(self.atn_dim)), std = 0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1 ), std=1)
    
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)
   
    def encode_observations(self, observations, state=None):
        ego_dim = 7
        partner_dim = 63 * 7
        road_dim = 200*7
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim:ego_dim+partner_dim]
        road_obs = observations[:, ego_dim+partner_dim:ego_dim+partner_dim+road_dim]
        
        partner_objects = partner_obs.view(-1, 63, 7)
        road_objects = road_obs.view(-1, 200, 7)
        road_continuous = road_objects[:, :, :6]  # First 6 features
        road_categorical = road_objects[:, :, 6]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, 200, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)
        
        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        
        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding
    
    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)
        return action, value

class Drone(nn.Module):
    ''' Drone policy. Flattens obs and applies a linear layer.
    '''
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space,
                pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict) 
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict) 

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            self.encoder = torch.nn.Sequential(
                nn.Linear(np.prod(env.single_observation_space.shape), hidden_size),
                nn.GELU(),
            )

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            self.decoder = pufferlib.pytorch.layer_init(
                    nn.Linear(hidden_size, sum(self.action_nvec)), std=0.01)
        elif not self.is_continuous:
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))

        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else: 
            observations = observations.view(batch_size, -1)
        return self.encoder(observations.float())

    def decode_actions(self, hidden):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values


class G2048(nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        num_obs = np.prod(env.single_observation_space.shape)

        if hidden_size <= 256:
            self.encoder = torch.nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(num_obs, 512)),
                nn.GELU(),
                pufferlib.pytorch.layer_init(nn.Linear(512, 256)),
                nn.GELU(),
                pufferlib.pytorch.layer_init(nn.Linear(256, hidden_size)),
                nn.GELU(),
            )
        else:
            self.encoder = torch.nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(num_obs, 2*hidden_size)),
                nn.GELU(),
                pufferlib.pytorch.layer_init(nn.Linear(2*hidden_size, hidden_size)),
                nn.GELU(),
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.GELU(),
            )

        num_atns = env.single_action_space.n
        self.decoder = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, num_atns), std=0.01),
        )
        self.value = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1).float()

        # Scale the feat 1 (tile**1.5)
        observations[:, :16] = observations[:, :16] / 100.0

        return self.encoder(observations)

    def decode_actions(self, hidden):
        logits = self.decoder(hidden)
        values = self.value(hidden)
        return logits, values


# ============================================================================
# Rymdboard Transformer Policy
# ============================================================================

class RymdboardTransformerBlock(nn.Module):
    """A single transformer encoder block with multi-head self-attention."""
    
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class Rymdboard(nn.Module):
    """
    Transformer-based policy for Rymdboard tile-placement game.
    
    Observation format (322 floats from C environment):
    - Board: 100 floats (normalized tile type + ownership per cell)
    - Market: 210 floats (7 cards × 30 features each)
    - Crystal: 12 floats (3 crystals × 4 features each)
    
    The policy reshapes the flat observation into tokens and processes
    with a transformer encoder.
    """
    
    def __init__(
        self, 
        env, 
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        num_layers=3,
        dropout=0.1,
        hidden_size=256,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.embed_dim = embed_dim
        
        # Observation dimensions (must match rymdboard.h)
        self.board_size = 100
        self.market_size = 210  # 7 cards × 30 features
        self.crystal_size = 12  # 3 crystals × 4 features
        self.num_cards = 7
        self.card_features = 30
        self.num_crystals = 3
        self.crystal_features = 4
        
        # Token embeddings
        self.board_embed = nn.Linear(1, embed_dim)  # Each cell is 1 float
        self.market_embed = nn.Linear(self.card_features, embed_dim)  # Each card is 30 features
        self.crystal_embed = nn.Linear(self.crystal_features, embed_dim)  # Each crystal is 4 features
        
        # Learnable type embeddings (board=0, market=1, crystal=2, cls=3)
        self.type_embed = nn.Embedding(4, embed_dim)
        
        # Learnable positional embeddings for board (10x10)
        self.board_pos_embed = nn.Parameter(torch.randn(1, 100, embed_dim) * 0.02)
        
        # Learnable positional embeddings for market (7 slots)
        self.market_pos_embed = nn.Parameter(torch.randn(1, 7, embed_dim) * 0.02)
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            RymdboardTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(embed_dim * 2, ff_dim)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(ff_dim, hidden_size)),
            nn.GELU(),
        )
        
        # Action space: 2800 discrete actions (7 cards × 10×10 positions × 4 rotations)
        num_actions = env.single_action_space.n
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value
    
    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values
    
    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        device = observations.device
        observations = observations.float()
        
        # Split observation into components
        board_obs = observations[:, :self.board_size]  # (B, 100)
        market_obs = observations[:, self.board_size:self.board_size + self.market_size]  # (B, 210)
        crystal_obs = observations[:, self.board_size + self.market_size:]  # (B, 12)
        
        # Embed board tokens (each cell as a token)
        board_tokens = board_obs.unsqueeze(-1)  # (B, 100, 1)
        board_tokens = self.board_embed(board_tokens)  # (B, 100, E)
        board_tokens = board_tokens + self.board_pos_embed
        board_types = torch.zeros(batch_size, 100, dtype=torch.long, device=device)
        board_tokens = board_tokens + self.type_embed(board_types)
        
        # Embed market tokens (7 cards, each with 30 features)
        market_tokens = market_obs.reshape(batch_size, self.num_cards, self.card_features)  # (B, 7, 30)
        market_tokens = self.market_embed(market_tokens)  # (B, 7, E)
        market_tokens = market_tokens + self.market_pos_embed
        market_types = torch.ones(batch_size, 7, dtype=torch.long, device=device)
        market_tokens = market_tokens + self.type_embed(market_types)
        
        # Embed crystal tokens (3 crystals, each with 4 features)
        crystal_tokens = crystal_obs.reshape(batch_size, self.num_crystals, self.crystal_features)  # (B, 3, 4)
        crystal_tokens = self.crystal_embed(crystal_tokens)  # (B, 3, E)
        crystal_types = torch.full((batch_size, 3), 2, dtype=torch.long, device=device)
        crystal_tokens = crystal_tokens + self.type_embed(crystal_types)
        
        # CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, E)
        cls_types = torch.full((batch_size, 1), 3, dtype=torch.long, device=device)
        cls_tokens = cls_tokens + self.type_embed(cls_types)
        
        # Concatenate all tokens: [CLS, Board, Market, Crystal]
        # Total: 1 + 100 + 7 + 3 = 111 tokens
        all_tokens = torch.cat([cls_tokens, board_tokens, market_tokens, crystal_tokens], dim=1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            all_tokens = layer(all_tokens)
        
        # Extract CLS token and mean of other tokens
        cls_output = all_tokens[:, 0, :]  # (B, E)
        mean_output = all_tokens[:, 1:, :].mean(dim=1)  # (B, E)
        
        # Combine and project
        combined = torch.cat([cls_output, mean_output], dim=1)  # (B, 2E)
        features = self.output_proj(combined)  # (B, hidden_size)
        
        return features
    
    def decode_actions(self, hidden):
        action = self.actor(hidden)
        value = self.value_fn(hidden)
        return action, value


class RymdboardLSTM(pufferlib.models.LSTMWrapper):
    """LSTM wrapper for Rymdboard transformer policy."""
    def __init__(self, env, policy, input_size=256, hidden_size=256):
        super().__init__(env, policy, input_size, hidden_size)


class SpatialCardEncoder(nn.Module):
    """Encodes the 5x5 shape of a market card using a tiny CNN."""
    def __init__(self, embed_dim):
        super().__init__()
        # Input: 1 channel (shape mask), 5x5 grid
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 5 * 5 + 3, embed_dim) # +3 for metadata (is_building, points, num_cells)

    def forward(self, card_obs):
        # card_obs: [Batch, 7 cards, 30 features]
        B, N, F_dim = card_obs.shape
        
        # Slice out features
        # [0] = is_building, [1] = points, [2] = num_cells
        metadata = card_obs[:, :, :3] 
        
        # [3:28] = 25 floats representing 5x5 grid
        shapes = card_obs[:, :, 3:28].reshape(B * N, 1, 5, 5) 
        
        # CNN Pass on shapes
        x = F.relu(self.conv1(shapes))
        x = F.relu(self.conv2(x))
        x = x.view(B, N, -1) # Flatten spatial dims
        
        # Concatenate metadata and project
        x = torch.cat([x, metadata], dim=2)
        return self.fc(x)

class RymdboardHybridPolicy(nn.Module):
    def __init__(
        self, 
        env, 
        embed_dim=128,
        num_heads=4,
        num_layers=2, # Reduced layers as CNN does heavy lifting
        hidden_size=256,
        **kwargs
    ):
        super().__init__()
        self.board_size = 100
        self.market_size = 210
        self.crystal_size = 12
        
        # --- 1. Board Encoder (Spatial) ---
        # We unpack the single float back into discrete categories
        self.type_embed = nn.Embedding(5, 32) # Empty, Street, Extraction, Base, Crystal
        self.owner_embed = nn.Embedding(2, 16) # Neutral, Player
        
        # CNN to process the 10x10 board naturally
        self.board_cnn = nn.Sequential(
            nn.Conv2d(32+16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
            # Output: [B, embed_dim, 10, 10] -> which we treat as 100 tokens of size embed_dim
        )

        # --- 2. Market Encoder (Spatial) ---
        self.market_encoder = SpatialCardEncoder(embed_dim)
        
        # --- 3. Transformer Backbone ---
        # We fuse Board (100 tokens) + Market (7 tokens)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 4. Heads ---
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(embed_dim, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(embed_dim, 1), std=1.0)

    def forward(self, observations, state=None):
        # We need to handle the PufferLib signature. 
        # Check if observations is a tuple (obs, mask) or just obs
        action_mask = None
        if isinstance(observations, tuple):
             observations, action_mask = observations
        elif observations.shape[-1] > 322:
             action_mask = observations[:, 322:]
             observations = observations[:, :322]
        
        # --- Preprocessing ---
        device = observations.device
        B = observations.shape[0]
        
        # Split Obs
        board_obs = observations[:, :100]
        market_obs = observations[:, 100:310].reshape(B, 7, 30)
        
        # --- 1. Process Board (Reverse Engineering C-Code) ---
        # Reversing: val = obs * 14.0
        # Logic: if val >= 10: owner=1, val-=10. else owner=0.
        raw_board = (board_obs * 14.0).round().long()
        
        # Extract ownership (>= 10 means owned)
        ownership = (raw_board >= 10).long()
        tile_type = raw_board % 10
        
        # Embed and shape to (B, C, H, W)
        emb_type = self.type_embed(tile_type).transpose(1, 2).view(B, 32, 10, 10)
        emb_owner = self.owner_embed(ownership).transpose(1, 2).view(B, 16, 10, 10)
        cnn_in = torch.cat([emb_type, emb_owner], dim=1)
        
        # CNN Pass
        board_features = self.board_cnn(cnn_in) # (B, 128, 10, 10)
        board_tokens = board_features.flatten(2).transpose(1, 2) # (B, 100, 128)
        
        # --- 2. Process Market ---
        market_tokens = self.market_encoder(market_obs) # (B, 7, 128)
        
        # --- 3. Fusion (Transformer) ---
        # Concatenate: [Board Tokens (100) | Market Tokens (7)]
        all_tokens = torch.cat([board_tokens, market_tokens], dim=1)
        
        # Self-Attention
        hidden = self.transformer(all_tokens)
        
        # Pooling (Global Max Pool usually works better than Mean for games)
        pooled = torch.max(hidden, dim=1)[0]
        
        # --- 4. Output ---
        actions = self.actor(pooled)
        value = self.value_fn(pooled)
        
        # --- CRITICAL: Action Masking ---
        if action_mask is not None:
            # Set invalid actions to a very large negative number
            actions = actions.masked_fill(action_mask == 0, -1e9)
            
        return actions, value
    
    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)