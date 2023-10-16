from torch import nn
import torch

from learn_bot.latent.transformer_nested_hidden_latent_model import combine_padding_sequence_masks

width = 10
num_heads = 2
num_layers = 4
transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=num_heads, batch_first=True)
transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers,
                                            enable_nested_tensor=False)
transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=width, nhead=num_heads, batch_first=True)
transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)
transformer_model = nn.Transformer(d_model=width, nhead=num_heads, num_encoder_layers=num_layers,
                                   num_decoder_layers=num_layers, custom_encoder=transformer_encoder, batch_first=True)

transformer_model.eval()
transformer_decoder.eval()
num_tokens = 4
in0 = torch.zeros(1, num_tokens, width)
in1 = in0.clone()
in1[0, 0, 0] = 1.
in2 = in1.clone()
in2[0, 1, 0] = 1.
in3 = in1.clone()
in3[0, 2, 0] = 1.

# diagonal mask, ensures players only look at themselves
src_tgt_mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
for i in range(num_tokens):
    for j in range(num_tokens):
        src_tgt_mask[i, j] = i != j
# disables player 0
key_mask = torch.BoolTensor([[i == 0 for i in range(num_tokens)]])
# testing that combination function preserves diagonal
combined_mask = combine_padding_sequence_masks(src_tgt_mask, key_mask, num_heads)
# disables players 2 and 3 for 0 and 1, vice versa
src_tgt_enemy_mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
for i in range(num_tokens):
    for j in range(num_tokens):
        src_tgt_enemy_mask[i, j] = (i // 2) != (j // 2)
# isolate players 0 and 1, combine 2 and 3 as 0 is dead, 1 is only one alive on team, 2 and 3 alive on same team
combined_enemy_mask = combine_padding_sequence_masks(src_tgt_enemy_mask, key_mask, num_heads)

print_matrix_values = False

enc_out0 = transformer_encoder(in0, mask=src_tgt_mask)
enc_out0_duplicate = transformer_encoder(in0, mask=src_tgt_mask)
enc_out1 = transformer_encoder(in1, mask=src_tgt_mask)
print("enc_out0 equals enc_out0_duplicate")
print(torch.equal(enc_out0, enc_out0_duplicate))
if print_matrix_values:
    print("enc_out0")
    print(enc_out0)
    print("enc_out1")
    print(enc_out1)
print("enc_out0 == enc_out1")
print(enc_out0 == enc_out1)
print(torch.sum(enc_out0 == enc_out1))

print("")

enc_enemy_out0 = transformer_encoder(in0, mask=combined_enemy_mask)
enc_enemy_out0_duplicate = transformer_encoder(in0, mask=combined_enemy_mask)
enc_enemy_out3 = transformer_encoder(in3, mask=combined_enemy_mask)
print("enc_enemy_out0 equals enc_enemy_out0_duplicate")
print(torch.equal(enc_enemy_out0, enc_enemy_out0_duplicate))
if print_matrix_values:
    print("enc_enemy_out0")
    print(enc_enemy_out0)
    print("enc_enemy_out3")
    print(enc_enemy_out3)
print("enc_enemy_out0 == enc_enemy_out3")
print(enc_enemy_out0 == enc_enemy_out3)
print(torch.sum(enc_enemy_out0 == enc_enemy_out3))

print("")

dec_out0 = transformer_decoder(in0, in0, tgt_mask=src_tgt_mask, memory_mask=src_tgt_mask)
dec_out0_duplicate = transformer_decoder(in0, in0, tgt_mask=src_tgt_mask, memory_mask=src_tgt_mask)
dec_out1 = transformer_decoder(in0, in1, tgt_mask=src_tgt_mask, memory_mask=src_tgt_mask)
print("dec_out0 equals dec_out0_duplicate")
print(torch.equal(dec_out0, dec_out0_duplicate))
if print_matrix_values:
    print("dec_out0")
    print(dec_out0)
    print("dec_out1")
    print(dec_out1)
print("dec_out0 == dec_out1")
print(dec_out0 == dec_out1)
print(torch.sum(dec_out0 == dec_out1))

print("")

enc_dec_out0 = transformer_model(in0, in0, src_mask=src_tgt_mask, tgt_mask=src_tgt_mask, memory_mask=src_tgt_mask)
enc_dec_out0_duplicate = transformer_model(in0, in0, src_mask=src_tgt_mask, tgt_mask=src_tgt_mask, memory_mask=src_tgt_mask)
enc_dec_out1 = transformer_model(in1, in0, src_mask=src_tgt_mask, tgt_mask=src_tgt_mask, memory_mask=src_tgt_mask)
print("enc_dec_out0 equals enc_dec_out0_duplicate")
print(torch.equal(enc_dec_out0, enc_dec_out0_duplicate))
if print_matrix_values:
    print("enc_dec_out0")
    print(enc_dec_out0)
    print("enc_dec_out1")
    print(enc_dec_out1)
print("enc_dec_out0 == enc_dec_out1")
print(enc_dec_out0 == enc_dec_out1)
print(torch.sum(enc_dec_out0 == enc_dec_out1))

print("")

key_enc_dec_out0 = transformer_model(in0, in0, src_mask=src_tgt_mask, tgt_key_padding_mask=key_mask,
                                     memory_key_padding_mask=key_mask)
key_enc_dec_out0_duplicate = transformer_model(in0, in0, src_mask=src_tgt_mask, tgt_key_padding_mask=key_mask,
                                               memory_key_padding_mask=key_mask)
key_enc_dec_out1 = transformer_model(in1, in0, src_mask=src_tgt_mask, tgt_key_padding_mask=key_mask,
                                     memory_key_padding_mask=key_mask)
print("key_enc_dec_out0 equals key_enc_dec_out0_duplicate")
print(torch.equal(key_enc_dec_out0, key_enc_dec_out0_duplicate))
if print_matrix_values:
    print("key_enc_dec_out0")
    print(key_enc_dec_out0)
    print("key_enc_dec_out1")
    print(key_enc_dec_out1)
print("key_enc_dec_out0 == key_enc_dec_out1")
print(key_enc_dec_out0 == key_enc_dec_out1)
print(torch.sum(key_enc_dec_out0 == key_enc_dec_out1))

#combined_mask = combined_mask & False
#combined_mask[:, :, 0] = True
combined_enc_out0 = transformer_encoder(in0, mask=combined_mask & False)
combined_enc_out0_duplicate = transformer_encoder(in0, mask=combined_mask)
combined_enc_out2 = transformer_encoder(in2, mask=combined_mask)
print("combined_enc_out0 equals combined_enc_out0_duplicate")
print(torch.equal(combined_enc_out0, combined_enc_out0_duplicate))
if print_matrix_values:
    print("combined_enc_out0")
    print(combined_enc_out0)
    print("combined_enc_out2")
    print(combined_enc_out2)
print("combined_enc_out0 == combined_enc_out1")
print(combined_enc_out0 == combined_enc_out2)
print(torch.sum(combined_enc_out0 == combined_enc_out2))

print("src_tgt_mask")
print(src_tgt_mask)
print("key_mask")
print(key_mask)
print("combined mask")
print(combined_mask)
print("src_tgt_enemy_mask")
print(src_tgt_enemy_mask)
print("combined enemy mask")
print(combined_enemy_mask)
