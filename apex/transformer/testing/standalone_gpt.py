import contextlib
import torch

from apex.transformer.enums import AttnMaskType
from apex.transformer import tensor_parallel

from apex.transformer.testing.global_vars import get_args
from apex.transformer.testing.standalone_transformer_lm import MegatronModule
from apex.transformer.testing.standalone_transformer_lm import parallel_lm_logits
from apex.transformer.testing.standalone_transformer_lm import post_language_model_processing
from apex.transformer.testing.standalone_transformer_lm import get_language_model
from apex.transformer.testing.standalone_transformer_lm import init_method_normal
from apex.transformer.testing.standalone_transformer_lm import (
    scaled_init_method_normal,
)



def gpt_model_provider(pre_process: bool = True, post_process: bool = True) -> "GPTModel":
    args = get_args()
    return GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        cpu_offload=args.cpu_offload,
    )


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(
        self,
        num_tokentypes:int = 0,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        cpu_offload: bool = False,
    ):
        super().__init__()
        args = get_args()

        self.forward_context = contextlib.nullcontext
        if cpu_offload:
            self.forward_context = torch.autograd.graph.save_on_cpu

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(
                args.init_method_std, args.num_layers
            ),
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        self.initialize_word_embeddings(init_method_normal)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        labels=None,
        tokentype_ids=None,
        inference_params=None,
    ):

        with self.forward_context():
            lm_output = self.language_model(
                input_ids, position_ids, attention_mask, inference_params=inference_params
            )

            if self.post_process:
                return post_language_model_processing(
                    lm_output,
                    labels,
                    self.word_embeddings_weight(),
                    self.parallel_output,
                    self.fp16_lm_cross_entropy,
                )
            else:
                return lm_output
