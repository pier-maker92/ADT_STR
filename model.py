import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ADTModelConfig
import torchaudio.transforms as T
from fast_transformers.masking import TriangularCausalMask, LengthMask
from fast_transformers.builders import TransformerDecoderBuilder, TransformerEncoderBuilder


def create_mask_plain(tgt_seq_len, tgt_lengths=None, device=None):
    tgt_mask = TriangularCausalMask(tgt_seq_len, device=device)
    if tgt_lengths is None:
        return tgt_mask
    tgt_padding_mask = LengthMask(torch.tensor(tgt_lengths), device=device)
    return tgt_mask, tgt_padding_mask


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
        mel_spec = self.compute_spec(wave)

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

        encoder_builder = TransformerEncoderBuilder()
        encoder_builder.n_layers = enc_layers
        encoder_builder.n_heads = nhead
        encoder_builder.feed_forward_dimensions = ffn_hid_dim
        encoder_builder.query_dimensions = d_query
        encoder_builder.value_dimensions = d_query
        encoder_builder.dropout = dropout
        encoder_builder.attention_type = "full"  # linear
        encoder_builder.attention_dropout = dropout
        self.encoder = encoder_builder.get()

    def forward(self, src_emb):
        src_emb = self.dense_layer(src_emb)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.dropout_layer(src_emb)
        encoder_output = self.encoder.forward(src_emb)
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

        decoder_builder = TransformerDecoderBuilder()
        decoder_builder.n_layers = dec_layers
        decoder_builder.n_heads = nhead
        decoder_builder.feed_forward_dimensions = ffn_hid_dim
        decoder_builder.query_dimensions = d_query
        decoder_builder.value_dimensions = d_query
        decoder_builder.dropout = dropout
        decoder_builder.attention_dropout = dropout
        decoder_builder.self_attention_type = "full"  # causal-linear
        decoder_builder.cross_attention_type = "full"  # linear
        self.decoder = decoder_builder.get()

    def forward(self, tgt, encoder_output, tgt_mask, tgt_padding_mask):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        tgt_emb = self.dropout_layer(tgt_emb)
        decoder_outputs = self.decoder.forward(
            tgt_emb,
            memory=encoder_output,
            x_mask=tgt_mask,
            x_length_mask=tgt_padding_mask,
            memory_mask=None,
            memory_length_mask=None,
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
        tgt_mask: torch.BoolTensor,
        tgt_padding_mask: torch.BoolTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        src_emb = self.compute_spectrogram(src)
        src_emb = self.project_to_mel(src_emb)
        encoder_out = self.encoder(src_emb)

        logits = self.decoder(
            tgt.float(),
            encoder_out.float(),
            tgt_mask,
            tgt_padding_mask,
        )
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
            # Create causal mask for current sequence length
            seq_len = generated.shape[1]
            tgt_mask, tgt_padding_mask = create_mask_plain(seq_len, seq_len, device)

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
