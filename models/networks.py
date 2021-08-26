import functools
import torch
from torch import nn

from utils.util import *


def get_norm_layer(norm_type='batch'):
    """Return a normalization layer.

    Parameters:
        norm_type (str) -- the name of the normalization layer: [batch | instance | none]
    """
    # functools.partial used to fix some parameters
    if norm_type == 'batch':
        # `track_running_stats=False` for unstable training in hiding and reveal tasks
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('Normalization layer [%s] is not found!' % norm_type)
    return norm_layer


def cat_zeros(X: torch.Tensor, input_nc: int):
    """Catenate `X` with a zero tensor to make its channel number equal to `input_nc`."""
    b, c, h, w = X.shape
    assert c <= input_nc, "The channel number of X (%d) is too large!" % c
    if c < input_nc:
        zeros = torch.zeros((b,input_nc-c,h,w)).to(X.device)
        X = torch.cat((X,zeros), dim=1)
    return X


def make_layers(block, num):
    """Make layers by repeating `block` `num` times."""
    layers = []
    for _ in range(num):
        layers.append(block)
    return nn.Sequential(*layers)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class Unet(nn.Module):
    """Create a U-Net."""
    def __init__(self, input_nc, output_nc, num_downs=5, nhf=64, norm_type='batch', use_dropout=False, output_function='tanh'):
        """
        Parameters:
            input_nc (int)        -- the number of channels in input images
            output_nc (int)       -- the number of channels in output images
            num_downs (int)       -- the number of downsamplings in UNet. For example, if |num_downs| == 7,
                                     image of size 128x128 will become of size 1x1 at the bottleneck
            nhf (int)             -- the number of filters in the last conv layer of hiding network
            norm_type (str)       -- normalization layer type
            use_dropout (bool)    -- if use dropout layers
            output_function (str) -- activation function for the outmost layer [sigmoid | tanh]

        We construct the U-Net from the innermost layer to the outermost layer, which is a recursive process.
        """
        super(Unet, self).__init__()
        self.input_nc = input_nc
        norm_layer = get_norm_layer(norm_type)

        # construct u-net strcture (from inner to outer)
        # add the innermost layer; nhf: number of filters in the last conv layer of hiding network
        unet_block = UnetSkipConnectionBlock(nhf*8, nhf*8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            # considering dropout
            unet_block = UnetSkipConnectionBlock(nhf*8, nhf*8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from nhf*8 to nhf
        unet_block = UnetSkipConnectionBlock(nhf*4, nhf*8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nhf*2, nhf*4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nhf, nhf*2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, nhf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_function=output_function)

        if output_function == 'tanh':
            self.factor = 10 / 255  # by referencing the engineering choice in universal adversarial perturbations
        elif output_function == 'sigmoid':
            self.factor = 1.0
        else:
            raise NotImplementedError('Activation funciton [%s] is not found!' % output_function)

    def forward(self, X):
        X = cat_zeros(X, self.input_nc)
        return self.factor * self.model(X)


class UnetSkipConnectionBlock(nn.Module):
    """Define the U-Net submodule with skip connection."""
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, output_function='tanh'):
        """
        Parameters:
            outer_nc (int)        -- the number of filters in the outer conv layer
            inner_nc (int)        -- the number of filters in the inner conv layer
            input_nc (int)        -- the number of channels in the input images/features
            submodule             -- previous defined submodules
            outermost (bool)      -- if this module is the outermost module
            innermost (bool)      -- if this module is the innermost module
            norm_layer            -- normalization layer
            use_dropout (bool)    -- if use dropout layers
            output_function (str) -- activation function for the outmost layer [sigmoid | tanh]
        """
        super(UnetSkipConnectionBlock, self).__init__()
        # `submodulde` is None <=> this block is an innermost block
        # `input_nc` is not None <=> this block is an outermost block

        self.outmost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)  # after Conv2d
        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(True)  # after ConvTranspose2d
        upnorm = norm_layer(outer_nc)

        if outermost:
            # no dropout; no relu in down; no norm in down & up
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            if output_function == 'tanh':
                up = [uprelu, upconv, nn.Tanh()]
            elif output_function == 'sigmoid':
                up = [uprelu, upconv, nn.Sigmoid()]
            else:
                raise NotImplementedError('Activation funciton [%s] is not found!' % output_function)
            model = down + [submodule] + up
        elif innermost:
            # no dropout; no norm in down
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outmost:
            return self.model(x)
        else:
            return torch.cat((x, self.model(x)), dim=1)  # cat by channel


class VanillaCNN(nn.Module):
    """Create a vanilla CNN."""
    def __init__(self, input_nc, output_nc, nrf=64, norm_type='batch', output_function='sigmoid'):
        """
        Parameters:
            input_nc (int)        -- the number of channels in the input images
            output_nc (int)       -- the number of channels in the output images
            nrf (int)             -- the number of filters in the last conv layer
            norm_type (str)       -- normalization layer type
            output_function (str) -- activation function for the last layer [sigmoid]
        """
        super(VanillaCNN, self).__init__()
        self.input_nc = input_nc

        # nrf: number of filters in the first/last conv layer of reveal network
        self.conv1 = nn.Conv2d(input_nc, nrf, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nrf, nrf*2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nrf*2, nrf*4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(nrf*4, nrf*2, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(nrf*2, nrf, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(nrf, output_nc, kernel_size=3, stride=1, padding=1)
        if output_function == 'sigmoid':
            self.output = nn.Sigmoid()
        else:
            raise NotImplementedError('Activation funciton [%s] is not found!' % output_function)
        
        self.relu = nn.ReLU(True)

        self.norm_layer = get_norm_layer(norm_type)
        self.norm1 = self.norm_layer(nrf)
        self.norm2 = self.norm_layer(nrf*2)
        self.norm3 = self.norm_layer(nrf*4)
        self.norm4 = self.norm_layer(nrf*2)
        self.norm5 = self.norm_layer(nrf)

    def forward(self, X):
        X = cat_zeros(X, self.input_nc)
        X = self.relu(self.norm1(self.conv1(X)))
        X = self.relu(self.norm2(self.conv2(X)))
        X = self.relu(self.norm3(self.conv3(X)))
        X = self.relu(self.norm4(self.conv4(X)))
        X = self.relu(self.norm5(self.conv5(X)))
        output = self.output(self.conv6(X))
        return output


class EncodingNet(nn.Module):
    """Create a fully connected layer for encoding keys."""
    def __init__(self, key_len, key_channel, redundance, batch_size):
        """
        Parameters:
            key_len (int)     -- length of the input key
            key_channel (int) -- channel number of the encoded key
            redundance (int)  -- redundance size of key; e.g. `32` for encoding key into a `key_channel`*32*32 tensor; `-1` for simple duplication
            batch_size (int)  -- batch size
        """
        super(EncodingNet, self).__init__()
        self.key_len, self.key_channel, self.redundance, self.batch_size = key_len, key_channel, redundance, batch_size
        if redundance != -1:
            self.linear = nn.Linear(key_len, key_channel*redundance*redundance)

    def forward(self, X):
        if self.redundance != -1:  # encode the input key by a fully connected layer
            X = self.linear(X)
            return X.view(1, self.key_channel, self.redundance, self.redundance).repeat(self.batch_size, 1, self.key_len//self.redundance, self.key_len//self.redundance)
        else:  # simply repeat
            return X.view(1, 1, 1, self.key_len).repeat(self.batch_size, self.key_channel, self.key_len, 1)


class DenseBlock(nn.Module):
    def __init__(self, input_nc, output_nc, nc=32, kaiming=False, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, nc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input_nc+nc, nc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input_nc+2*nc, nc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input_nc+3*nc, nc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input_nc+4*nc, output_nc, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        init_inv_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1, kaiming)
        init_inv_weights(self.conv5, 0, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5


class InvBlock(nn.Module):
    def __init__(self, input_nc, split_nc, n=3):
        super(InvBlock, self).__init__()

        self.nc1, self.nc2 = split_nc, input_nc-split_nc

        if n > 1:
            assert self.nc1 == self.nc2, "Number of nput and output channel should be the same when using multiple blocks!"
        self.F = make_layers(DenseBlock(self.nc2, self.nc1), n)
        self.G = make_layers(DenseBlock(self.nc1, self.nc2), n)
        self.H = make_layers(DenseBlock(self.nc1, self.nc2), n)

    def forward(self, x, rev=False):
        x1, x2 = x[:, :self.nc1, :, :], x[:, self.nc1:, :, :]
        if not rev:
            y1 = x1 + self.F(x2)
            y2 = x2.mul(torch.exp(torch.sigmoid(self.H(y1))*2 - 1)) + self.G(y1)
        else:
            y2 = (x2 - self.G(x1)).div(torch.exp(torch.sigmoid(self.H(x1))*2 - 1))
            y1 = x1 - self.F(y2)
        return torch.cat((y1, y2), dim=1)


class InvHidingNet(nn.Module):
    def __init__(self, input_nc, split_nc, N=8, n=3):
        super(InvHidingNet, self).__init__()
        operations = []
        for _ in range(N):
            block = InvBlock(input_nc, split_nc, n)
            operations.append(block)
        self.model_list = nn.ModuleList(operations)

    def forward(self, X, rev=False):
        if not rev:
            for m in self.model_list:
                output = m.forward(X, rev)
        else:
            for m in reversed(self.model_list):
                output = m.forward(X, rev)
        return output
