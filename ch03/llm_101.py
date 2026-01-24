#Layer Normalization...
## Why ??
###  Training Deep Neural Network with many layers can be challeningi due to 
####    1) Vanishing Gradient Problem and 
####    2) Exporing Gradient Problem
### Layer Normalization helps to stabilize and accelerate the training process by normalizing the inputs across the features for each data point independently.
### It ensures that the inputs to each layer have a consistent distribution, which helps in maintaining stable gradients during backpropagation.
#### Improve Stability and Efficiency of Neural Network Training.
##### If the Gradient is too small the learning will stagnate.
##### And if the gradient is too large the learning will be unstable.
##### Batch Normalization helps to keep the gradients in a reasonable range and stable.
##### Another benefit of Layer Normalization is that it can help to reduce the sensitivity of the model to the initial weights.
##### That is - it prevents internal covariate shift. 
##### Which implies that the distribution of inputs to a layer changes during training as the parameters of the previous layers change is not a problem anymore.
##### That means that each layer can learn on a more stable distribution of inputs, which can lead to faster convergence and better generalization.
##### Done by keeping the means (Mean = 0)  and standard deviations (std dev = 1) of the inputs to each layer consistent throughout training.

import torch 
import torch.nn as nn

class DummyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        #The parameters here are just to mimic the LayerNorm Interface. 

    def forward(self, x):
        return x
    

torch.manual_seed(123)
batch_example = torch.randn(2,5)
print("Input Tensor:\n", batch_example)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)
print("Output Tensor before LayerNorm:\n", out)

mean = out.mean(dim=-1, keepdim=True)
variance = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", variance)
normalized_out = (out - mean) / torch.sqrt(variance + 1e-5)
print("Output Tensor after Manual Normalization:\n", normalized_out)
normalized_out.mean(dim=-1), normalized_out.std(dim=-1)
print("Mean after Normalization:\n", normalized_out.mean(dim=-1))
normalized_out.var(dim=-1)
print("Variance after Normalization:\n", normalized_out.var(dim=-1))

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2) / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
           nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), #allows for exploring more complex patterns
           #nn.ReLU(),
           GeLU(),
           nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


GPT_CONFIG_124M = {
    "emb_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "vocab_size": 50257,
    "max_seq_len": 1024,
    "drop_rate": 0.1,
    "context_length": 1024,
    "qkv_bias": True,
}


print (GPT_CONFIG_124M["emb_dim"])

ffn = FeedForward(GPT_CONFIG_124M)
dummy_input = torch.randn(2, 5, GPT_CONFIG_124M["emb_dim"])
ffn_output = ffn(dummy_input)
print("FeedForward Output Shape:", ffn_output.shape)

print("Input: ", dummy_input[0,0,:5])
print("FeedForward Output: ", ffn_output[0,0,:5])

class ExampleDeepNeuralNetwork(nn.Module): 
    def __init__(self,layersizes,use_shortcut=True):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layersizes[0], layersizes[1]), nn.ReLU()),
            nn.Sequential(nn.Linear(layersizes[1], layersizes[2]), nn.ReLU()),
            nn.Sequential(nn.Linear(layersizes[2], layersizes[3]), nn.ReLU()),
            nn.Sequential(nn.Linear(layersizes[3], layersizes[4]), nn.ReLU()),
            nn.Sequential(nn.Linear(layersizes[4], layersizes[5]), nn.ReLU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute output of current layer
            layer_output = layer(x)
            # Apply residual connection if use_shortcut is True
            if self.use_shortcut:
                x = x + layer_output  # Residual connection
            else:
                x = layer_output
        return x
    
layer_sizes = [3,3,3,3,3,1]
sample_input = torch.tensor([[1.,0.,-1.]])
torch.manual_seed(123)
output_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)(sample_input)
print("Output without Shortcut: ", output_without_shortcut)

output_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)(sample_input)
print("Output with Shortcut: ", output_with_shortcut)


def printGradients(model, x):
      
    # forward pass
    output = model(x)
    target = torch.tensor([[0.]])
    
    #calculate the loss based on how close the target is to the output
    loss = nn.MSELoss()(output, target)
    
    #backward pass
    loss.backward()
      
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the wieghts
            print(f"Mean Absolute Gradient for {name}: {torch.mean(torch.abs(param.grad))}")
            
printGradients(ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True), sample_input)
printGradients(ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False), sample_input)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        print(f"Initializing MultiHeadAttention with d_in={d_in}, d_out={d_out}, num_heads={num_heads}")
        assert (d_out % num_heads == 0 ) # d_out must be divisible by num_heads
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim
        
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key   = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out,d_out) # Linear Layer to combine head outputs
        self.register_buffer(
            "mask",torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
       
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)

        attn_scores = torch.matmul(queries, keys.transpose(-2,-1)) / (self.head_dim ** 0.5)

        attn_scores = attn_scores.masked_fill(self.mask[:num_tokens,:num_tokens].bool(), -torch.inf)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.transpose(1,2).contiguous().view(b, num_tokens, self.d_out)

        output = self.out_proj(attn_output)

        return output
    
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads = cfg["num_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        
    def forward(self,x):
        #shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut

        #shortcut connection for feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3

print("batch.shape:", batch.shape)  # Should be (1, 6,
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


dat = nn.Linear(4,3) 
print("Linear Data " , dat.weight)
print("Linear Data Shape " , dat.weight.shape)
print("Linear Bias " ,dat.bias)

i = dat(torch.randn(2,4))
print(i.shape)
print(i)

t = TransformerBlock(GPT_CONFIG_124M)
dummy_input = torch.randn(2, 10, GPT_CONFIG_124M["emb_dim"])
t_output = t(dummy_input)
print("Transformer Block Output Shape:", t_output.shape)
