import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from types import SimpleNamespace
from .utils import download_ckpt
from .config import Config
from netdissect import proggan, zdataset
from . import biggan
from . import stylegan
from . import stylegan2
from abc import abstractmethod, ABC as AbstractBaseClass
from functools import singledispatch

class BaseModel(AbstractBaseClass, torch.nn.Module):

    def __init__(self, model_name, class_name):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.outclass = class_name

    @abstractmethod
    def partial_forward(self, x, layer_name):
        pass

    @abstractmethod
    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        pass

    def get_max_latents(self):
        return 1

    def latent_space_name(self):
        return 'Z'

    def get_latent_shape(self):
        return tuple(self.sample_latent(1).shape)

    def get_latent_dims(self):
        return np.prod(self.get_latent_shape())

    def set_output_class(self, new_class):
        self.outclass = new_class

    def forward(self, x):
        out = self.model.forward(x)
        return 0.5*(out+1)

    def sample_np(self, z=None, n_samples=1, seed=None):
        if z is None:
            z = self.sample_latent(n_samples, seed=seed)
        elif isinstance(z, list):
            z = [torch.tensor(l).to(self.device) if not torch.is_tensor(l) else l for l in z]
        elif not torch.is_tensor(z):
            z = torch.tensor(z).to(self.device)
        img = self.forward(z)
        img_np = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        return np.clip(img_np, 0.0, 1.0).squeeze()

    def get_conditional_state(self, z):
        return None

    def set_conditional_state(self, z, c):
        return z

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)

# Define Deep Autoencoder
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, encoding_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(latents, encoding_dim, num_epochs=1000, batch_size=32, learning_rate=1e-3):
    autoencoder = DeepAutoencoder(latents.shape[1], encoding_dim).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(latents))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for data in dataloader:
            img = data[0].cuda()
            encoded, decoded = autoencoder(img)
            loss = criterion(decoded, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return autoencoder

class StyleGAN2(BaseModel):
    def __init__(self, device, class_name, truncation=1.0, use_w=False):
        super(StyleGAN2, self).__init__('StyleGAN2', class_name or 'ffhq')
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = use_w  # use W as primary latent space?
        self.autoencoder = None  # Initialize autoencoder attribute

        # Image widths
        configs = {
            # Converted NVIDIA official
            'ffhq': 1024,
            'car': 512,
            'cat': 256,
            'church': 256,
            'horse': 256,
            # Tuomas
            'bedrooms': 256,
            'kitchen': 256,
            'places': 256,
            'lookbook': 512
        }

        assert self.outclass in configs, \
            f'Invalid StyleGAN2 class {self.outclass}, should be one of [{", ".join(configs.keys())}]'

        self.resolution = configs[self.outclass]
        self.name = f'StyleGAN2-{self.outclass}'
        self.has_latent_residual = True
        self.load_model()
        self.set_noise_seed(0)

    def latent_space_name(self):
        return 'W' if self.w_primary else 'Z'

    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False

    def download_checkpoint(self, outfile):
        checkpoints = {
            'horse': 'https://drive.google.com/uc?export=download&id=18SkqWAkgt0fIwDEf2pqeaenNi4OoCo-0',
            'ffhq': 'https://drive.google.com/uc?export=download&id=1FJRwzAkV-XWbxgTwxEmEACvuqF5DsBiV',
            'church': 'https://drive.google.com/uc?export=download&id=1HFM694112b_im01JT7wop0faftw9ty5g',
            'car': 'https://drive.google.com/uc?export=download&id=1iRoWclWVbDBAy5iXYZrQnKYSbZUqXI6y',
            'cat': 'https://drive.google.com/uc?export=download&id=15vJP8GDr0FlRYpE8gD7CdeEz2mXrQMgN',
            'places': 'https://drive.google.com/uc?export=download&id=1X8-wIH3aYKjgDZt4KMOtQzN1m4AlCVhm',
            'bedrooms': 'https://drive.google.com/uc?export=download&id=1nZTW7mjazs-qPhkmbsOLLA_6qws-eNQu',
            'kitchen': 'https://drive.google.com/uc?export=download&id=15dCpnZ1YLAnETAPB0FGmXwdBclbwMEkZ',
            'lookbook': 'https://drive.google.com/uc?export=download&id=1-F-RMkbHUv_S_k-_olh43mu5rDUMGYKe'
        }

        url = checkpoints[self.outclass]
        download_ckpt(url, outfile)

    def load_model(self):
        checkpoint_root = os.environ.get('GANCONTROL_CHECKPOINT_DIR', Path(__file__).parent / 'checkpoints')
        checkpoint = Path(checkpoint_root) / f'stylegan2/stylegan2_{self.outclass}_{self.resolution}.pt'

        self.model = stylegan2.Generator(self.resolution, 512, 8).to(self.device)

        if not checkpoint.is_file():
            os.makedirs(checkpoint.parent, exist_ok=True)
            self.download_checkpoint(checkpoint)

        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['g_ema'], strict=False)
        self.latent_avg = 0

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)  # use (reproducible) global rand state

        rng = np.random.RandomState(seed)
        z = torch.from_numpy(
            rng.standard_normal(512 * n_samples)
                .reshape(n_samples, 512)).float().to(self.device)  # [N, 512]

        if self.w_primary:
            z = self.model.style(z)

        return z

    def get_max_latents(self):
        return self.model.n_latent

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError('StyleGAN2: cannot change output class without reloading')

    def forward(self, x):
        x = x if isinstance(x, list) else [x]
        out, _ = self.model(x, noise=self.noise,
                            truncation=self.truncation, truncation_latent=self.latent_avg, input_is_w=self.w_primary)
        return 0.5 * (out + 1)

    def partial_forward(self, x, layer_name):
        styles = x if isinstance(x, list) else [x]
        inject_index = None
        noise = self.noise

        if not self.w_primary:
            styles = [self.model.style(s) for s in styles]

        if len(styles) == 1:
            inject_index = self.model.n_latent
            latent = self.model.strided_style(styles[0].unsqueeze(1).repeat(1, inject_index, 1))  # [N, 18, 512]
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.model.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.model.n_latent - inject_index, 1)

            latent = self.model.strided_style(torch.cat([latent, latent2], 1))
        else:
            assert len(styles) == self.model.n_latent, f'Expected {self.model.n_latents} latents, got {len(styles)}'
            styles = torch.stack(styles, dim=1)  # [N, 18, 512]
            latent = self.model.strided_style(styles)

        if 'style' in layer_name:
            return

        out = self.model.input(latent)
        if 'input' == layer_name:
            return

        out = self.model.conv1(out, latent[:, 0], noise=noise[0])
        if 'conv1' in layer_name:
            return

        skip = self.model.to_rgb1(out, latent[:, 1])
        if 'to_rgb1' in layer_name:
            return

        i = 1
        noise_i = 1

        for conv1, conv2, to_rgb in zip(
                self.model.convs[::2], self.model.convs[1::2], self.model.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise[noise_i])
            if f'convs.{i - 1}' in layer_name:
                return

            out = conv2(out, latent[:, i + 1], noise=noise[noise_i + 1])
            if f'convs.{i}' in layer_name:
                return

            skip = to_rgb(out, latent[:, i + 2], skip)
            if f'to_rgbs.{i // 2}' in layer_name:
                return

            i += 2
            noise_i += 2

        image = skip

        raise RuntimeError(f'Layer {layer_name} not encountered in partial_forward')

    def set_noise_seed(self, seed):
        torch.manual_seed(seed)
        self.noise = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=self.device)]

        for i in range(3, self.model.log_size + 1):
            for _ in range(2):
                self.noise.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=self.device))

    # Additional methods to integrate autoencoder
    def train_autoencoder(self, latents, encoding_dim, num_epochs=1000, batch_size=32, learning_rate=1e-3):
        self.autoencoder = train_autoencoder(latents, encoding_dim, num_epochs, batch_size, learning_rate)

    def encode_latents(self, latents):
        latents = torch.FloatTensor(latents).cuda()
        with torch.no_grad():
            encoded, _ = self.autoencoder(latents)
        return encoded.cpu().numpy()

    def decode_latents(self, encoded_latents):
        encoded_latents = torch.FloatTensor(encoded_latents).cuda()
        with torch.no_grad():
            _, decoded = self.autoencoder(encoded_latents)
        return decoded.cpu().numpy()

    def manipulate_latent(self, latent, principal_component, factor):
        encoded_latent = self.encode_latents([latent])[0]
        manipulated_encoded_latent = encoded_latent + factor * principal_component
        manipulated_latent = self.decode_latents([manipulated_encoded_latent])[0]
        return manipulated_latent
