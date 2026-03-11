import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ADTModelConfig
import torchaudio.transforms as T
from utils.utils import create_mask_plain

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        """
        Args:
            vocab_size: Vocabulary size
            emb_size: Dimension of embedding vectors
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, token_vector: torch.Tensor):
        """
        Args:
            token_vector: One-hot or multi-hot tensor (batch, seq_len, vocab_size)
        Returns:
            Embedded vectors (batch, seq_len, emb_size)
        """
        # Get embedding weights (vocab_size, emb_size)
        embedding_weights = self.embedding.weight  # Get weights

        # Multiply one-hot or multi-hot vector with embedding weights
        embeddings = torch.matmul(token_vector, embedding_weights) * math.sqrt(self.emb_size)
        
        return embeddings
    
class TokenEmbedding_plain(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding_plain, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 maxlen: int = 2048):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return token_embedding + self.pos_embedding[:, :token_embedding.size(1), :]

class ComputeMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate, win_length, time_res, n_mels, device):
        super(ComputeMelSpectrogram, self).__init__()
        self.device = device
        self.compute_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length,
            hop_length=int(time_res * sample_rate),
            n_mels=n_mels,
            f_min=20.0,
            power=2,
        ).to(device)
        self.window_pad_idxs = int((win_length / 2) // int(time_res * sample_rate) + 1)

    def forward(self, wave):
        wave = wave.to(self.device)
        # Esegui in float32: torchaudio MelSpectrogram + bf16 autocast causano CUBLAS_STATUS_INVALID_VALUE
        with torch.amp.autocast(device_type="cuda", enabled=False):
            wave_f32 = wave.float()
            mel_spec = self.compute_spec(wave_f32)

        logmel_spec = torch.log(mel_spec + 1e-10)
        logmel_spec = torch.clamp(logmel_spec, -23, 12)
        logmel_spec = (logmel_spec + 23) / (12 + 23)

        return logmel_spec.permute(0, 2, 1)[
            :, self.window_pad_idxs : -(self.window_pad_idxs + 1), :
        ]  # (batch, len, emb)


class Encoder(nn.Module):
    def __init__(
        self,
        enc_layers,
        d_query,
        nhead,
        ffn_hid_dim,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.num_features = d_query * nhead
        self.dense_layer = nn.Linear(self.num_features, self.num_features, bias=False)
        self.positional_encoding = PositionalEncoding(self.num_features)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.num_features, elementwise_affine=True)
        self.dropout_layer = nn.Dropout(p=dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=self.num_features,
            nhead=nhead,
            dim_feedforward=ffn_hid_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=enc_layers)

    def forward(self, src_emb):
        src_emb = self.dense_layer(src_emb)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.dropout_layer(src_emb)
        encoder_output = self.encoder(src_emb)
        encoder_output = self.dropout_layer(self.layer_norm(encoder_output))
        return encoder_output


class Decoder(nn.Module):
    def __init__(self, dec_layers, d_query, nhead, ffn_hid_dim, tgt_vocab_size, dropout, plain=False):
        super(Decoder, self).__init__()
        self.num_features = d_query * nhead
        if plain:
            self.tgt_tok_emb = TokenEmbedding_plain(tgt_vocab_size, self.num_features)
        if not plain:
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, self.num_features)
        self.positional_encoding = PositionalEncoding(self.num_features)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.generator = nn.Linear(self.num_features, tgt_vocab_size)

        layer = nn.TransformerDecoderLayer(
            d_model=self.num_features,
            nhead=nhead,
            dim_feedforward=ffn_hid_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=dec_layers)

    def forward(self, tgt, encoder_output, tgt_mask, tgt_padding_mask):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        tgt_emb = self.dropout_layer(tgt_emb)
        # bf16-safe and PyTorch same-type: use additive float masks (0 / -1e4) instead of bool (-inf)
        if tgt_mask is not None:
            tgt_mask = torch.zeros_like(tgt_mask, dtype=tgt_emb.dtype, device=tgt_mask.device).masked_fill_(tgt_mask, -1e4)
        if tgt_padding_mask is not None and tgt_padding_mask.dtype == torch.bool:
            tgt_padding_mask = torch.zeros_like(tgt_padding_mask, dtype=tgt_emb.dtype, device=tgt_padding_mask.device).masked_fill_(tgt_padding_mask, -1e4)
        decoder_outputs = self.decoder(
            tgt_emb,
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_mask=None,
            memory_key_padding_mask=None,
        )
        return self.generator(decoder_outputs)


class ADTModel(nn.Module):
    def __init__(
        self,
        config: ADTModelConfig,
        compute_spectrogram: ComputeMelSpectrogram,
    ) -> None:
        super(ADTModel, self).__init__()
        self.config = config
        self.encoder = Encoder(
            enc_layers=config.enc_layers,
            d_query=config.d_query,
            nhead=config.nhead,
            ffn_hid_dim=int(config.d_query * config.nhead * 4),
            dropout=config.dropout,
        )
        self.decoder = Decoder(
            dec_layers=config.dec_layers,
            d_query=config.d_query,
            nhead=config.nhead,
            ffn_hid_dim=int(config.d_query * config.nhead * 4),
            dropout=config.dropout,
            tgt_vocab_size=config.tgt_vocab_size,
            plain=config.plain,
        )
        self.compute_spectrogram = compute_spectrogram
        self.project_to_mel = nn.Linear(config.n_mels, int(config.d_query * config.nhead))

    def _loss_fn(self, logits: torch.FloatTensor, tgt: torch.LongTensor) -> torch.FloatTensor:
        # Use fp32 for stable loss and ignore PAD token (id=1)
        logits = logits.float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt.long().reshape(-1),
            ignore_index=1,
        )

    def forward(
        self,
        src: torch.FloatTensor,
        tgt: torch.LongTensor,
        tgt_mask: Optional[torch.BoolTensor],
        tgt_padding_mask: Optional[torch.BoolTensor],
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        src_emb = self.compute_spectrogram(src)
        src_emb = self.project_to_mel(src_emb)
        encoder_out = self.encoder(src_emb)

        # Build causal mask inside model so DataParallel does not split (T,T) across devices
        if tgt_mask is None:
            tgt_seq_len = tgt.size(1)
            tgt_mask, _ = create_mask_plain(tgt_seq_len, None, tgt.device)
        logits = self.decoder(tgt, encoder_out, tgt_mask, tgt_padding_mask)
        loss = self._loss_fn(logits, labels)
        return loss

    def sample(
        self,
        src: torch.FloatTensor,
        src_mask: torch.BoolTensor,
        tgt_mask: torch.BoolTensor,
        max_length: int = 1000,
        start_token: int = 2,
        end_token: int = 3,
    ) -> torch.LongTensor:
        """
        Basic autoregressive greedy sampling for ADT model.

        Args:
            src: Input audio tensor (batch_size, audio_length)
            src_mask: Source mask (not used in current implementation)
            tgt_mask: Target mask (not used in current implementation)
            max_length: Maximum sequence length to generate
            start_token: Token ID to start generation (default: 0)
            end_token: Token ID to end generation (default: 2, assuming EOS)

        Returns:
            Generated token sequence (batch_size, seq_length)
        """
        if not self.config.plain:
            raise NotImplementedError("Non-plain mode is not implemented")
        self.eval()
        device = src.device
        batch_size = src.shape[0]

        # Process input through encoder
        src_emb = self.compute_spectrogram(src)
        src_emb = self.project_to_mel(src_emb)
        memory = self.encoder(src_emb)

        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        # Greedy decoding loop
        for step in range(max_length - 1):
            # Create causal mask for current sequence length (no padding in decoding)
            seq_len = generated.shape[1]
            tgt_mask, tgt_padding_mask = create_mask_plain(seq_len, None, device)

            # Get decoder output
            logits = self.decoder(
                generated,
                memory,
                tgt_mask,
                tgt_padding_mask=None,
            )

            # Get next token (greedy)
            next_token_logits = logits[:, -1, :]  # Last position
            next_token = torch.argmax(next_token_logits, dim=-1)  # Greedy selection

            # Check for end token
            if torch.all(next_token == end_token):
                break

            # Append next token
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        return generated

    def beam_search(
        self,
        src: torch.FloatTensor,
        src_mask: torch.BoolTensor,
        tgt_mask: torch.BoolTensor,
        beam_size: int = 5,
        max_length: int = 1000,
        start_token: int = 0,
        end_token: int = 2,
        length_penalty: float = 1.0,
    ) -> torch.LongTensor:
        """
        Beam search sampling for ADT model.

        Args:
            src: Input audio tensor (batch_size, audio_length)
            src_mask: Source mask (not used in current implementation)
            tgt_mask: Target mask (not used in current implementation)
            beam_size: Number of beams to keep at each step
            max_length: Maximum sequence length to generate
            start_token: Token ID to start generation (default: 0)
            end_token: Token ID to end generation (default: 2, assuming EOS)
            length_penalty: Length penalty factor for scoring (default: 1.0)

        Returns:
            Best generated token sequence (batch_size, seq_length)
        """
        if not self.config.plain:
            raise NotImplementedError("Non-plain mode is not implemented")

        self.eval()
        device = src.device
        batch_size = src.shape[0]

        # Process input through encoder
        src_emb = self.compute_spectrogram(src)
        src_emb = self.project_to_mel(src_emb)
        memory = self.encoder(src_emb)

        # Initialize beams for each batch
        # Each beam contains: [batch_idx, sequence, log_prob, finished]
        all_beams = []

        for batch_idx in range(batch_size):
            # Initialize beams for this batch item
            initial_sequence = torch.full((1, 1), start_token, dtype=torch.long, device=device)
            beams = [{"sequence": initial_sequence, "log_prob": 0.0, "finished": False}]
            all_beams.append(beams)

        # Beam search loop
        for step in range(max_length - 1):
            new_all_beams = []

            for batch_idx in range(batch_size):
                beams = all_beams[batch_idx]
                new_beams = []

                # Skip if all beams are finished
                if all(beam["finished"] for beam in beams):
                    new_all_beams.append(beams)
                    continue

                # Collect all active sequences for this batch
                active_beams = [beam for beam in beams if not beam["finished"]]
                if not active_beams:
                    new_all_beams.append(beams)
                    continue

                # Stack sequences for parallel processing
                sequences = torch.cat([beam["sequence"] for beam in active_beams], dim=0)
                batch_memory = memory[batch_idx : batch_idx + 1].expand(len(active_beams), -1, -1)

                # Create causal mask
                seq_len = sequences.shape[1]
                tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

                # Get decoder output
                logits = self.decoder(
                    sequences,
                    batch_memory,
                    tgt_mask,
                    None,
                )

                # Get next token probabilities
                next_token_logits = logits[:, -1, :]  # Last position
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

                # Generate candidates for each active beam
                candidates = []
                for beam_idx, beam in enumerate(active_beams):
                    beam_log_probs = log_probs[beam_idx]
                    top_k_log_probs, top_k_tokens = torch.topk(beam_log_probs, beam_size)

                    for k in range(beam_size):
                        token = top_k_tokens[k].item()
                        token_log_prob = top_k_log_probs[k].item()
                        new_log_prob = beam["log_prob"] + token_log_prob

                        # Create new sequence
                        new_sequence = torch.cat([beam["sequence"], torch.tensor([[token]], device=device)], dim=1)

                        # Check if finished
                        finished = token == end_token

                        candidates.append({"sequence": new_sequence, "log_prob": new_log_prob, "finished": finished})

                # Add finished beams (they don't expand further)
                finished_beams = [beam for beam in beams if beam["finished"]]
                candidates.extend(finished_beams)

                # Select top beam_size candidates
                candidates.sort(key=lambda x: self._score_sequence(x, length_penalty), reverse=True)
                new_beams = candidates[:beam_size]

                new_all_beams.append(new_beams)

            all_beams = new_all_beams

            # Check if all beams in all batches are finished
            all_finished = all(all(beam["finished"] for beam in beams) for beams in all_beams)
            if all_finished:
                break

        # Select best sequence for each batch
        results = []
        for batch_idx in range(batch_size):
            beams = all_beams[batch_idx]
            # Sort by score and take the best
            beams.sort(key=lambda x: self._score_sequence(x, length_penalty), reverse=True)
            best_sequence = beams[0]["sequence"]
            results.append(best_sequence)

        # Stack results and pad to same length
        max_len = max(seq.shape[1] for seq in results)
        padded_results = []
        for seq in results:
            if seq.shape[1] < max_len:
                padding = torch.full((1, max_len - seq.shape[1]), end_token, dtype=torch.long, device=device)
                padded_seq = torch.cat([seq, padding], dim=1)
            else:
                padded_seq = seq
            padded_results.append(padded_seq)

        return torch.cat(padded_results, dim=0)

    def _score_sequence(self, beam: dict, length_penalty: float) -> float:
        seq_len = beam["sequence"].shape[1]
        if length_penalty == 0.0:
            return beam["log_prob"]
        else:
            # Length normalization: log_prob / (length^length_penalty)
            return beam["log_prob"] / (seq_len**length_penalty)
