"""
This is file which will assist to help us translate the text which can be used to overcome imbalanced data
To run this file one should install these packages tensorflow.compat.v1, tensorflow_hub, sentencepiece, transformers
"""

'''
!pip install transformers -U -q
!pip install sentencepiece

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import sentencepiece
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
'''


model_name = 'facebook/mbart-large-50-many-to-many-mmt'

model = MBartForConditionalGeneration.from_pretrained(model_name)

encoder_tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX")
decoder_tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ta_IN")

# article_en = ['U.N encourages wearing masks','My name is khan I am not a terrorist']


def encoder_text(text):
  model_inputs = encoder_tokenizer(text, return_tensors="pt")
  generated_tokens = model.generate(**model_inputs,forced_bos_token_id=encoder_tokenizer.lang_code_to_id["ta_IN"])
  translation = encoder_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
  return translation[0]

def decoder_text(text):
  model_inputs = decoder_tokenizer(text, return_tensors="pt")
  generated_tokens = model.generate(**model_inputs,forced_bos_token_id=decoder_tokenizer.lang_code_to_id["en_XX"])
  translation = decoder_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
  return translation[0]