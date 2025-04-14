import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.core import BaseModel


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

# Protect imports used in other scripts
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, compression_factor):
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv1d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1),
        )
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(num_hiddens, embedding_dim, kernel_size=1, stride=1)

    def forward(self, x, compression_factor):
        x = x.view([x.shape[0], 1, x.shape[-1]])
        x = self._conv(x)
        x = self._residual_stack(x)
        return self._pre_vq_conv(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, compression_factor):
        super().__init__()
        self._conv_1 = nn.Conv1d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._deconv = nn.Sequential(
            nn.ConvTranspose1d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(num_hiddens // 2, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x, compression_factor):
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = self._deconv(x)
        return x.squeeze()


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        input_shape = x.shape
        flat_input = x.view(-1, self._embedding_dim)
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, self._embedding.weight, encoding_indices, encodings


class vqvae(BaseModel):
    def __init__(self, vqvae_config):
        super().__init__()
        num_hiddens = vqvae_config['block_hidden_size']
        num_residual_layers = vqvae_config['num_residual_layers']
        num_residual_hiddens = vqvae_config['res_hidden_size']
        embedding_dim = vqvae_config['embedding_dim']
        num_embeddings = vqvae_config['num_embeddings']
        commitment_cost = vqvae_config['commitment_cost']
        self.compression_factor = vqvae_config['compression_factor']

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.encoder = Encoder(1, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, self.compression_factor)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, self.compression_factor)

    def shared_eval(self, batch, optimizer, mode, comet_logger=None):
        if mode == 'train':
            optimizer.zero_grad()
            z = self.encoder(batch, self.compression_factor)
            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
            data_recon = self.decoder(quantized, self.compression_factor)
            recon_error = F.mse_loss(data_recon, batch)
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()

        elif mode in ['val', 'test']:
            with torch.no_grad():
                z = self.encoder(batch, self.compression_factor)
                vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
                data_recon = self.decoder(quantized, self.compression_factor)
                recon_error = F.mse_loss(data_recon, batch)
                loss = recon_error + vq_loss

        if comet_logger is not None:
            comet_logger.log_metric(f'{mode}_vqvae_loss_each_batch', loss.item())
            comet_logger.log_metric(f'{mode}_vqvae_vq_loss_each_batch', vq_loss.item())
            comet_logger.log_metric(f'{mode}_vqvae_recon_loss_each_batch', recon_error.item())
            comet_logger.log_metric(f'{mode}_vqvae_perplexity_each_batch', perplexity.item())

        return loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings


