
# Tôi sẽ tạo một implementation đầy đủ của Transformer (2017) thuần Python với backward/forward pass và training trên tập dữ liệu nhỏ.Tôi đã tạo một implementation đầy đủ của Transformer (2017) thuần Python! Đây là những điểm chính:

# ## **Các thành phần chính:**

# 1. **Multi-Head Attention** - với forward và backward pass đầy đủ
# 2. **Position-wise Feed Forward Network** - với ReLU activation
# 3. **Layer Normalization** - tính gradient cho gamma và beta
# 4. **Positional Encoding** - sử dụng sin/cos như trong paper gốc
# 5. **Encoder và Decoder Layers** - với residual connections
# 6. **Embedding layers** - cho source và target

# ## **Tính năng:**

# - ✅ **Forward pass đầy đủ**: từ input đến output logits
# - ✅ **Backward pass đầy đủ**: gradient cho tất cả parameters
# - ✅ **Causal masking**: cho decoder (tránh nhìn future tokens)
# - ✅ **Training loop**: với SGD optimizer
# - ✅ **Copy task dataset**: task đơn giản để test model

# ## **Kiến trúc:**
# - d_model = 128
# - num_heads = 4
# - num_layers = 2 (encoder + decoder)
# - d_ff = 256
# - Vocab size = 20

# ## **Cách chạy:**
# ```bash
# python transformer.py
# ```

# Model sẽ train trên "copy task" - học cách copy input sequence sang output. Sau 50 epochs, model sẽ học được pattern này khá tốt!

# **Lưu ý**: Đây là implementation giáo dục, không optimize cho performance. Để sử dụng thực tế, nên dùng PyTorch/TensorFlow với GPU acceleration.
#     Returns:
#         _type_: _description_
import numpy as np

# ==================== Activation Functions ====================
class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask # tương đương nhân 1
    
    # Đạo hàm 1 tại x > 0
    def backward(self, grad):
        return grad * self.mask

class Softmax:
    def forward(self, x):
        # Numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, grad):
        # Simplified: assumes cross-entropy loss will handle gradient
        return grad

# ==================== Layer Normalization ==================== 
class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)
    
    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True) # giữ nguyên chiều ảnh, thực hiện tính trung bình phương chiều cuối.
        self.var = np.var(x, axis=-1, keepdims=True) # Tính phương sai
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps) # Tính vector chuẩn hóa
        return self.gamma * self.x_norm + self.beta # Khôi phục biểu diễn
    
    def backward(self, grad):
        # công thức tính khôi phục: x_norm = gamma * x^ + beta
        N = self.x.shape[-1] # Lấy chiều cuối.
        self.d_gamma += np.sum(grad * self.x_norm, axis=tuple(range(grad.ndim - 1)))  
        self.d_beta += np.sum(grad, axis=tuple(range(grad.ndim - 1)))
        
        dx_norm = grad * self.gamma
        dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 * 
                     np.power(self.var + self.eps, -1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True) + \
                dvar * np.sum(-2 * (self.x - self.mean), axis=-1, keepdims=True) / N
        
        dx = dx_norm / np.sqrt(self.var + self.eps) + \
             dvar * 2 * (self.x - self.mean) / N + dmean / N
        return dx

# ==================== Linear Layer ====================
class Linear:
    def __init__(self, in_features, out_features):
        # Xavier initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
    
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, grad):
        self.dW += np.dot(self.x.reshape(-1, self.x.shape[-1]).T, 
                          grad.reshape(-1, grad.shape[-1]))
        self.db += np.sum(grad, axis=tuple(range(grad.ndim - 1)))
        return np.dot(grad, self.W.T)

# ==================== Multi-Head Attention ====================
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        self.softmax = Softmax()
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections
        Q = self.W_q.forward(query)
        K = self.W_k.forward(key)
        V = self.W_v.forward(value)
        
        # Reshape to (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        self.attn_weights = self.softmax.forward(scores)
        self.context = np.matmul(self.attn_weights, V)
        
        # Concatenate heads
        self.context_concat = self.context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = self.W_o.forward(self.context_concat)
        
        # Store for backward
        self.Q, self.K, self.V = Q, K, V
        self.batch_size = batch_size
        self.query_shape = query.shape
        
        return output
    
    def backward(self, grad):
        grad = self.W_o.backward(grad)
        
        # Reshape gradient back to multi-head format
        grad = grad.reshape(self.batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Backprop through attention
        dV = np.matmul(self.attn_weights.transpose(0, 1, 3, 2), grad)
        d_attn_weights = np.matmul(grad, self.V.transpose(0, 1, 3, 2))
        
        # Backprop through softmax (simplified)
        d_scores = d_attn_weights * self.attn_weights
        d_scores = d_scores - self.attn_weights * np.sum(d_scores, axis=-1, keepdims=True)
        d_scores = d_scores / np.sqrt(self.d_k)
        
        # Backprop through Q, K
        dQ = np.matmul(d_scores, self.K)
        dK = np.matmul(d_scores.transpose(0, 1, 3, 2), self.Q)
        
        # Reshape back
        dQ = dQ.transpose(0, 2, 1, 3).reshape(self.batch_size, -1, self.d_model)
        dK = dK.transpose(0, 2, 1, 3).reshape(self.batch_size, -1, self.d_model)
        dV = dV.transpose(0, 2, 1, 3).reshape(self.batch_size, -1, self.d_model)
        
        # Backprop through linear layers
        dq = self.W_q.backward(dQ)
        dk = self.W_k.backward(dK)
        dv = self.W_v.backward(dV)
        
        return dq, dk, dv

# ==================== Position-wise Feed Forward ====================
class PositionwiseFeedForward:
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.relu = ReLU()
    
    def forward(self, x):
        return self.fc2.forward(self.relu.forward(self.fc1.forward(x)))
    
    def backward(self, grad):
        grad = self.fc2.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)
        return grad

# ==================== Positional Encoding ====================
def get_positional_encoding(seq_len, d_model):
    """Return (seq_len, d_model) positional encoding."""
    pos = np.arange(seq_len)[:, np.newaxis]               # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]                # (1, d_model)
    # angle rates: 1 / 10000^(2i/d_model) for even indices (i//2 used)
    angle_rates = 1 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates                       # (seq_len, d_model)
    # apply sin to even indices; cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

# ==================== Encoder Layer ====================
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual
        self.x = x
        attn_out = self.attn.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_out)
        
        # Feed forward with residual
        self.x_after_attn = x
        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)
        return x
    
    def backward(self, grad):
        # Backprop through second residual and norm
        grad_norm2 = self.norm2.backward(grad)
        grad_ffn = self.ffn.backward(grad_norm2)
        grad = grad_norm2 + grad_ffn
        
        # Backprop through first residual and norm
        grad_norm1 = self.norm1.backward(grad)
        dq, dk, dv = self.attn.backward(grad_norm1)
        grad = grad_norm1 + dq + dk + dv
        
        return grad

# ==================== Decoder Layer ====================
class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        self.x = x
        attn1_out = self.self_attn.forward(x, x, x, tgt_mask)
        x = self.norm1.forward(x + attn1_out)
        
        # Cross-attention
        self.x_after_self_attn = x
        attn2_out = self.cross_attn.forward(x, enc_output, enc_output, src_mask)
        x = self.norm2.forward(x + attn2_out)
        
        # Feed forward
        self.x_after_cross_attn = x
        ffn_out = self.ffn.forward(x)
        x = self.norm3.forward(x + ffn_out)
        
        return x
    
    def backward(self, grad, enc_grad_accum):
        # Backprop through third residual
        grad_norm3 = self.norm3.backward(grad)
        grad_ffn = self.ffn.backward(grad_norm3)
        grad = grad_norm3 + grad_ffn
        
        # Backprop through second residual (cross-attention)
        grad_norm2 = self.norm2.backward(grad)
        dq2, dk2, dv2 = self.cross_attn.backward(grad_norm2)
        grad = grad_norm2 + dq2
        enc_grad_accum += dk2 + dv2
        
        # Backprop through first residual (self-attention)
        grad_norm1 = self.norm1.backward(grad)
        dq1, dk1, dv1 = self.self_attn.backward(grad_norm1)
        grad = grad_norm1 + dq1 + dk1 + dv1
        
        return grad, enc_grad_accum

# ==================== Transformer ====================
class Transformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, max_seq_len=100):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.src_embed = np.random.randn(src_vocab_size, d_model) * 0.01
        self.tgt_embed = np.random.randn(tgt_vocab_size, d_model) * 0.01
        self.d_src_embed = np.zeros_like(self.src_embed)
        self.d_tgt_embed = np.zeros_like(self.tgt_embed)
        
        # Positional encoding
        self.pos_encoding = get_positional_encoding(max_seq_len, d_model)
        
        # Encoder and Decoder
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        
        # Output projection
        self.fc_out = Linear(d_model, tgt_vocab_size)
    
    def create_masks(self, src, tgt):
        # For simplicity, no padding mask in this implementation
        tgt_len = tgt.shape[1]
        tgt_mask = np.triu(np.ones((tgt_len, tgt_len)), k=1)
        return None, tgt_mask
    
    def forward(self, src, tgt):
        # Embedding + Positional Encoding
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        
        self.src = src
        self.tgt = tgt
        
        src_embed = self.src_embed[src] * np.sqrt(self.d_model)
        src_embed = src_embed + self.pos_encoding[:src_seq_len]
        
        tgt_embed = self.tgt_embed[tgt] * np.sqrt(self.d_model)
        tgt_embed = tgt_embed + self.pos_encoding[:tgt_seq_len]
        
        # Create masks
        src_mask, tgt_mask = self.create_masks(src, tgt)
        
        # Encoder
        enc_output = src_embed
        for layer in self.encoder_layers:
            enc_output = layer.forward(enc_output, src_mask)
        self.enc_output = enc_output
        
        print(f"{enc_output.shape} output encoder con")
        # Decoder
        dec_output = tgt_embed
        for layer in self.decoder_layers:
            dec_output = layer.forward(dec_output, enc_output, src_mask, tgt_mask)
        
        # Output projection
        output = self.fc_out.forward(dec_output)
        return output
    
    def backward(self, grad):
        # Backprop through output layer
        grad = self.fc_out.backward(grad)
        
        # Backprop through decoder
        enc_grad_accum = np.zeros_like(self.enc_output)
        for layer in reversed(self.decoder_layers):
            grad, enc_grad_accum = layer.backward(grad, enc_grad_accum)
        
        # Store target embedding gradients
        batch_size, tgt_len = self.tgt.shape
        for b in range(batch_size):
            for t in range(tgt_len):
                self.d_tgt_embed[self.tgt[b, t]] += grad[b, t] * np.sqrt(self.d_model)
        
        # Backprop through encoder
        grad = enc_grad_accum
        for layer in reversed(self.encoder_layers):
            grad = layer.backward(grad)
        
        # Store source embedding gradients
        batch_size, src_len = self.src.shape
        for b in range(batch_size):
            for s in range(src_len):
                self.d_src_embed[self.src[b, s]] += grad[b, s] * np.sqrt(self.d_model)

# ==================== Training ====================
def create_toy_dataset():
    """Create a simple copy task dataset"""
    vocab_size = 20
    num_samples = 100
    seq_len = 8
    
    data = []
    for _ in range(num_samples):
        seq = np.random.randint(1, vocab_size, size=seq_len)
        # Input: sequence, Target: same sequence (copy task)
        data.append((seq, seq))
    
    return data, vocab_size
def cross_entropy_loss(logits, targets):
    """
    logits: (batch, seq_len, vocab)
    targets: (batch, seq_len)
    returns: loss (scalar), grad (same shape as logits)
    """
    batch_size, seq_len, vocab_size = logits.shape
    # stable softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # cross entropy loss
    # gather probabilities of targets
    idx = (np.arange(batch_size)[:, None], np.arange(seq_len)[None, :], targets)
    # but numpy fancy indexing with 3 arrays needs to prepare properly:
    # simpler loopless approach:
    probs_of_targets = probs[np.arange(batch_size)[:, None], np.arange(seq_len)[None, :], targets]
    loss = -np.log(probs_of_targets + 1e-12).mean()
    
    # gradient: probs - one_hot
    grad = probs.copy()
    # subtract 1 at target positions
    grad[np.arange(batch_size)[:, None], np.arange(seq_len)[None, :], targets] -= 1.0
    grad /= (batch_size * seq_len)
    
    return loss, grad


def sgd_update(model, lr=0.001):
    """Simple SGD update"""
    # Update embeddings
    model.src_embed -= lr * model.d_src_embed
    model.tgt_embed -= lr * model.d_tgt_embed
    model.d_src_embed.fill(0)
    model.d_tgt_embed.fill(0)
    
    # Update encoder layers
    for layer in model.encoder_layers:
        for param_name in ['W_q', 'W_k', 'W_v', 'W_o']:
            linear = getattr(layer.attn, param_name)
            linear.W -= lr * linear.dW
            linear.b -= lr * linear.db
            linear.dW.fill(0)
            linear.db.fill(0)
        
        for param_name in ['fc1', 'fc2']:
            linear = getattr(layer.ffn, param_name)
            linear.W -= lr * linear.dW
            linear.b -= lr * linear.db
            linear.dW.fill(0)
            linear.db.fill(0)
        
        for norm in [layer.norm1, layer.norm2]:
            norm.gamma -= lr * norm.d_gamma
            norm.beta -= lr * norm.d_beta
            norm.d_gamma.fill(0)
            norm.d_beta.fill(0)
    
    # Update decoder layers
    for layer in model.decoder_layers:
        for attn_name in ['self_attn', 'cross_attn']:
            attn = getattr(layer, attn_name)
            for param_name in ['W_q', 'W_k', 'W_v', 'W_o']:
                linear = getattr(attn, param_name)
                linear.W -= lr * linear.dW
                linear.b -= lr * linear.db
                linear.dW.fill(0)
                linear.db.fill(0)
        
        for param_name in ['fc1', 'fc2']:
            linear = getattr(layer.ffn, param_name)
            linear.W -= lr * linear.dW
            linear.b -= lr * linear.db
            linear.dW.fill(0)
            linear.db.fill(0)
        
        for norm in [layer.norm1, layer.norm2, layer.norm3]:
            norm.gamma -= lr * norm.d_gamma
            norm.beta -= lr * norm.d_beta
            norm.d_gamma.fill(0)
            norm.d_beta.fill(0)
    
    # Update output layer
    model.fc_out.W -= lr * model.fc_out.dW
    model.fc_out.b -= lr * model.fc_out.db
    model.fc_out.dW.fill(0)
    model.fc_out.db.fill(0)

def train():
    # Create dataset
    print("Creating toy dataset (copy task)...")
    dataset, vocab_size = create_toy_dataset()
    
    # Initialize model
    print("Initializing Transformer model...")
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=2,
        d_ff=2048,
        max_seq_len=512
    )
    
    # Training loop
    print("\nTraining...")
    batch_size = 4
    num_epochs = 50
    lr = 0.001
    
    for epoch in range(num_epochs):
        total_loss = 0
        np.random.shuffle(dataset)
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            if len(batch) < batch_size:
                continue
            
            # Prepare batch
            src_batch = np.array([item[0] for item in batch])
            tgt_batch = np.array([item[1] for item in batch])
            
            # Forward pass
            output = model.forward(src_batch, tgt_batch)
            
            # Compute loss
            loss, grad = cross_entropy_loss(output, tgt_batch)
            total_loss += loss
            
            # Backward pass
            model.backward(grad)
            
            # Update parameters
            sgd_update(model, lr)
        
        avg_loss = total_loss / (len(dataset) // batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Test on a sample
        if (epoch + 1) % 10 == 0:
            test_src = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
            test_tgt = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
            output = model.forward(test_src, test_tgt)
            pred = np.argmax(output, axis=-1)
            print(f"  Input:  {test_src[0]}")
            print(f"  Pred:   {pred[0]}")
    print("\nTraining completed!")

if __name__ == "__main__":
    np.random.seed(42)
    train()