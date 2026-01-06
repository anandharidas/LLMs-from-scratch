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

class SelfAttention_v1(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(SelfAttention_v1, self).__init__()
        self.W__query = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False)   # Weight matrix for query
        self.W__key   = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False)   # Weight matrix for key
        self.W__value = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False) # Weight matrix for value

    def forward(self, inputs):
        query = inputs @ self.W__query   # Transformed query for all inputs
        keys = inputs @ self.W__key     # Transformed keys for all inputs
        values = inputs @ self.W__value # Transformed values for all inputs

        attn_scores = query @ keys.T  # Attention scores for query against all keys
        attn_weights = torch.softmax(attn_scores / (keys.shape[1] ** 0.5), dim=-1)  # Softmax to get attention weights
        values = attn_weights @ values  # Weighted sum of values based on attention weights 
        return attn_scores,values
    
class SelfAttention_v2(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(SelfAttention_v2, self).__init__()
        self.W__query = torch.nn.Linear(d_in, d_out, bias=False)   # Weight matrix for query
        self.W__key   = torch.nn.Linear(d_in, d_out, bias=False)   # Weight matrix for key
        self.W__value = torch.nn.Linear(d_in, d_out, bias=False) # Weight matrix for value

    def forward(self, inputs):
        query = self.W__query(inputs)   # Transformed query for all inputs
        keys = self.W__key(inputs)     # Transformed keys for all inputs
        values = self.W__value(inputs) # Transformed values for all inputs

        attn_scores = query @ keys.T  # Attention scores for query against all keys
        attn_weights = torch.softmax(attn_scores / (keys.shape[1] ** 0.5), dim=-1)  # Softmax to get attention weights
        values = attn_weights @ values  # Weighted sum of values based on attention weights 
        return attn_scores,values
    
    
d_in = inputs.shape[1] # the input embedding size , d = 3
d_out = 2 # the output embedding size , d' = 2  
torch.manual_seed(123) # for reproducibility
self_attn = SelfAttention_v1(d_in, d_out)
attn_scores, values = self_attn(inputs)
print("Attention Scores (using original query and transformed keys):\n", attn_scores)
print("Values after applying attention weights:\n", values)


torch.manual_seed(789) # for reproducibility
self_attn = SelfAttention_v2(d_in, d_out)
attn_scores, values = self_attn(inputs)
print("Attention Scores (using original query and transformed keys):\n", attn_scores)
print("Values after applying attention weights:\n", values)


context_length = inputs.shape[0]
result = torch.triu(torch.ones(context_length, context_length), diagonal=1)  # Upper triangular matrix
print("Upper Triangular Matrix:\n", result)

masked = attn_scores.masked_fill(result.bool(), -torch.inf)  # Apply mask to attention scores
print("Masked Attention Scores:\n", masked)
attn_weights = torch.softmax(masked / (d_out ** 0.5), dim=-1)  # Softmax to get attention weights
print("Attention Weights after Masking:\n", attn_weights)
torch.manual_seed(123) # for reproducibility
dropout_layer = torch.nn.Dropout(p=0.5)
attn_weights_dropped = dropout_layer(attn_weights)
print("Attention Weights after Dropout:\n", attn_weights_dropped)
values_dropped = attn_weights_dropped @ values  # Weighted sum of values based on dropped attention weights 
print("Values after applying attention weights with Dropout:\n", values_dropped)