import torch
from importlib.metadata import version

print("Torch Version: ", version("torch"))

inputs = torch.tensor(
   [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1] # 2nd input token is the query (x^2)

attn_scores_2 = torch.empty(inputs.shape[0]) # to store attention scores
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, x_i) # dot product attention
    
print("Attention Scores (for query 'journey'):\n", attn_scores_2)

x2 = inputs[1] # 2nd input element
d_in = inputs.shape[1] # the input embedding size , d = 3
d_out = 2 # the output embedding size , d' = 2

torch.manual_seed(123) # for reproducibility

W__query = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False)   # Weight matrix for query
W__key   = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False)   # Weight matrix for key
W__value = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False) # Weight matrix for value
print("W__query:\n", W__query)
print("W__key:\n", W__key)
print("W__value:\n", W__value)  

query_2 = x2 @ W__query   # Transformed query
key_2   = x2 @ W__key     # Transformed key
value_2 = x2 @ W__value   # Transformed value   

print("Query_2: " , query_2)

keys = inputs @ W__key     # Transformed keys for all inputs
values = inputs @ W__value # Transformed values for all inputs

print("keys.shape: ", keys.shape)       # (6, 2)
print("values.shape: ", values.shape)   # (6, 2)

keys_2 = keys[1]       # 2nd input's transformed key
print("keys_2: ", keys_2)
attn_scores_2 = query_2 @ keys_2.T  # Attention score for query_2 against key_2
print("Attention Score (using transformed query and key_2):\n", attn_scores_2)

attn_scores_22 = query_2 @ keys.T  # Attention scores for query_2 against all keys
print("Attention Scores (using transformed query and keys):\n", attn_scores_22)


query = inputs @ W__query   # Transformed query for all inputs
keys = inputs @ W__key     # Transformed keys for all inputs
attn_scores = query @ keys.T  # Attention scores for query_2 against all keys
print("Attention Scores (using original query and transformed keys):\n", attn_scores)