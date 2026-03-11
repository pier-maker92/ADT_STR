import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoTokenizer
from transformers import ClapModel, ClapProcessor


class ClapWrapper(nn.Module):
    def __init__(self, model_name:str, device:torch.device, sample_rate:int, **kwargs):
        super().__init__()
        self.clap_model = ClapModel.from_pretrained(model_name)
        self.device = device
        self.clap_model.to(device)
        self.sample_rate = sample_rate
        self.config = self.clap_model.config
        self.text_model = self.clap_model.text_model
        self.audio_model = self.clap_model.audio_model
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_audio_features(self, audios):
        audios = [audio.squeeze(0).numpy() for audio in audios]
        inputs = self.processor(audio=audios, return_tensors="pt", sampling_rate=self.sample_rate).to(self.device)
        return self._get_audio_features(**inputs)

    def get_text_features(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return self._get_text_features(**inputs)

    def _get_audio_features(
        self,
        input_features: Optional[torch.Tensor] = None,
        is_longer: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
        ) -> torch.FloatTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            return_dict=return_dict,
        )

        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_features = self.clap_model.audio_projection(pooled_output)

        return audio_features / audio_features.norm(p=2, dim=-1, keepdim=True)

    def _get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # Use CLAP model's config for some fields (if specified) instead of those of audio & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1] if return_dict is not None else text_outputs.pooler_output
        text_features = self.clap_model.text_projection(pooled_output)

        return text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    def semantic_match(self, audio_features:torch.FloatTensor, text_features:torch.FloatTensor):
        # cosine similarity as logits
        logit_scale_audio = self.clap_model.logit_scale_a.exp()
        logits_per_audio = torch.matmul(audio_features, text_features.t()) * logit_scale_audio
        return logits_per_audio.softmax(dim=-1)