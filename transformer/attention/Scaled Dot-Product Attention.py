import numpy as np
import matplotlib.pyplot as plt

class ScaledDotProductAttention:
    """
    Implementation cơ chế Scaled Dot-Product Attention nguyên bản
    Công thức: Attention(Q,K,V) = softmax(QK^T/√d_k)V 
    """
    # Chia cho √d_k để ổn định gradient
    def __init__(self, d_k):
        """
        d_k: Dimensionality của key vectors
        """
        self.d_k = d_k
        self.attention_weights = None # Trọng số attention
        
    def forward(self, Q, K, V, mask=None):
        """
            Q: Query matrix [batch_size, seq_len_q, d_k]
            K: Key matrix [batch_size, seq_len_k, d_k]
            V: Value matrix [batch_size, seq_len_v, d_v]
            mask: Optional mask [batch_size, seq_len_q, seq_len_k]
            
            Returns:
            output: [batch_size, seq_len_q, d_v]
            attention_weights: [batch_size, seq_len_q, seq_len_k]
        """
        
        batch_size = Q.shape[0]
        seq_len_q = Q.shape[1]
        seq_len_k = K.shape[1]
        
        # Bước 1: Tính điểm số (scores) = QK^T
        # Q: [batch_size, seq_len_q, d_k]
        # K^T: [batch_size, d_k, seq_len_k]
        # scores: [batch_size, seq_len_q, seq_len_k]
        scores = np.matmul(Q, K.transpose(0, 2, 1))
        print(f"Scores shape: {scores.shape}")
        print(f"Raw scores (trước scaling):\n{scores[0][:3, :3]}\n")
        
        # Bước 2: Scale scores bằng √d_k
        scores = scores / np.sqrt(self.d_k)
        print(f"Scaled scores (sau khi chia cho √{self.d_k}):\n{scores[0][:3, :3]}\n")
        print(f"Hallo {scores.shape}")
        # Bước 3: Áp dụng mask nếu có (cho masked attention)
        if mask is not None:
            # Set vị trí bị mask thành -inf để sau softmax sẽ thành 0
            scores = np.where(mask == 0, -1e9, scores)
            print(f"Scores sau khi apply mask:\n{scores[0][:3, :3]}\n")
        
        # Bước 4: Áp dụng softmax để có attention weights
        self.attention_weights = self.softmax(scores, axis=-1)
        print(f"hsa {self.attention_weights.shape}")
        print(f"Attention weights (sau softmax):\n{self.attention_weights[0][:3, :3]}\n")
        
        # Bước 5: Nhân attention weights với Value matrix
        # attention_weights: [batch_size, seq_len_q, seq_len_k]
        # V: [batch_size, seq_len_v, d_v]
        # output: [batch_size, seq_len_q, d_v]
        output = np.matmul(self.attention_weights, V)
        print(f"Output shape: {output.shape}")
        
        return output, self.attention_weights
    
    def softmax(self, x, axis=-1):
        """Softmax function implementation"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism
    """
    
    def __init__(self, d_model, num_heads):
        """
        d_model: Dimensionality của model
        num_heads: Số lượng attention heads
        """
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # d_k số chiều mỗi head
        # Linear projections cho Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
        self.attention = ScaledDotProductAttention(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        """
        query, key, value: [batch_size, seq_len, d_model]
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Bước 1: Linear projections
        Q = np.matmul(query, self.W_q)  # [batch_size, seq_len, d_model]
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)
        
        # Bước 2: Reshape thành multi-heads
        # [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Reshape để apply attention
        Q = Q.reshape(-1, seq_len, self.d_k)
        K = K.reshape(-1, seq_len, self.d_k)
        V = V.reshape(-1, seq_len, self.d_k)
        
        if mask is not None:
            mask = np.repeat(mask, self.num_heads, axis=0)
        
        # Bước 3: Apply scaled dot-product attention
        # Trả về context vector và attention weight
        attn_output, attn_weights = self.attention.forward(Q, K, V, mask)
        
        # Bước 4: Reshape và concatenate heads
        attn_output = attn_output.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Bước 5: Final linear projection
        output = np.matmul(attn_output, self.W_o)
        
        # Reshape attention weights để visualization
        attn_weights = attn_weights.reshape(batch_size, self.num_heads, seq_len, seq_len)
        
        # trả về kết quả sau lớp FC, và trọng số multi head attention
        return output, attn_weights


def create_causal_mask(seq_len):
    """Tạo causal mask cho autoregressive attention (decoder)"""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask

def visualize_attention(attention_weights, tokens=None):
    """Visualize attention weights dưới dạng heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Nếu có multi-heads, lấy trung bình
    if len(attention_weights.shape) == 4:
        # [batch_size, num_heads, seq_len, seq_len] -> [seq_len, seq_len]
        attention_weights = attention_weights[0].mean(axis=0)
    elif len(attention_weights.shape) == 3:
        # [batch_size, seq_len, seq_len] -> [seq_len, seq_len]
        attention_weights = attention_weights[0]
    
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    if tokens:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_yticklabels(tokens)
    
    ax.set_xlabel('Keys')
    ax.set_ylabel('Queries')
    ax.set_title('Attention Weights Visualization')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


# ============ DEMO VÀ TEST ============

def demo_basic_attention():
    """Demo cơ chế attention cơ bản"""
    print("="*50)
    print("DEMO 1: BASIC SCALED DOT-PRODUCT ATTENTION")
    print("="*50)
    
    # Thiết lập parameters
    batch_size = 1
    seq_len = 4
    
    # Mỗi token(từ) khi được biến đổi thành các Key Vector
    d_k = 8
    
    # Mỗi token cx sẽ có 1 value vector
    d_v = 8
    
    # Tạo data giả, 3 ma trận giống nhau
    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, d_k) # query: đặt câu hỏi rằng từ này ở trong câu có ý nghĩa gì
    K = np.random.randn(batch_size, seq_len, d_k) # key: trả lời rằng từ hiện tại có liên quan thế nào đến từ mà query đang hỏi
    V = np.random.randn(batch_size, seq_len, d_v) # value: chứa thông tin của từ đang hỏi
    
    # Khởi tạo attention
    attention = ScaledDotProductAttention(d_k)
    
    # Forward pass
    output, weights = attention.forward(Q, K, V)
    
    print(f"\nInput shapes:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Visualize
    tokens = ['Token1', 'Token2', 'Token3', 'Token4']
    visualize_attention(weights, tokens)


def demo_masked_attention():
    """Demo masked attention (cho decoder)"""
    print("\n" + "="*50)
    print("DEMO 2: MASKED (CAUSAL) ATTENTION")
    print("="*50)
    
    batch_size = 1
    seq_len = 5
    d_k = 8
    d_v = 8
    
    # Tạo data
    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)
    
    # Tạo causal mask
    mask = create_causal_mask(seq_len)
    mask = np.expand_dims(mask, 0)  # Add batch dimension
    
    print(f"Causal mask:\n{mask[0]}\n")
    
    # Apply attention với mask
    attention = ScaledDotProductAttention(d_k)
    output, weights = attention.forward(Q, K, V, mask)
    
    print(f"Masked attention weights (chú ý tam giác dưới):")
    print(weights[0])
    
    # Visualize
    tokens = [f'Tok{i+1}' for i in range(seq_len)]
    visualize_attention(weights, tokens)

def demo_multi_head_attention():
    """Demo Multi-Head Attention"""
    print("\n" + "="*50)
    print("DEMO 3: MULTI-HEAD ATTENTION")
    print("="*50)
    
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8
    
    # Tạo data
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Khởi tạo multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Self-attention: Q=K=V=x
    output, weights = mha.forward(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"  (batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, seq_len={seq_len})")
    
    # Visualize attention của head đầu tiên
    tokens = [f'Pos{i+1}' for i in range(seq_len)]
    visualize_attention(weights[:, 0:1, :, :], tokens)

def example_text_attention():
    """Ví dụ với text thực tế (simplified)"""
    print("\n" + "="*50)
    print("DEMO 4: TEXT ATTENTION EXAMPLE")
    print("="*50)
    
    # Giả lập word embeddings
    words = ["The", "cat", "sat", "on", "mat"]
    seq_len = len(words)
    d_model = 16
    
    # Random embeddings (thực tế sẽ từ embedding layer)
    np.random.seed(42)
    embeddings = np.random.randn(1, seq_len, d_model) # 1 x 5 x 16
    
    # Apply self-attention
    mha = MultiHeadAttention(d_model, num_heads=4) # mỗi từ sẽ được embedding thành 1 vector 16 chiều, trải qua lớp linear trở thành vector 4 chiều
    output, weights = mha.forward(embeddings, embeddings, embeddings)
    
    print(f"Words: {words}")
    print(f"Input embeddings shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    
    # Visualize attention pattern
    visualize_attention(weights, words)
    
    # Phân tích attention scores
    avg_weights = weights[0].mean(axis=0)  # Average across heads
    print("\nAttention analysis (averaged across heads):")
    for i, word_i in enumerate(words):
        print(f"\n'{word_i}' attends to:")
        for j, word_j in enumerate(words):
            score = avg_weights[i, j]
            print(f"  '{word_j}': {score:.3f}")

if __name__ == "__main__":
    # Chạy các demo
    # demo_basic_attention()
    # demo_masked_attention()
    # demo_multi_head_attention()
    example_text_attention()
    print("\n" + "="*50)
    print("KẾT THÚC DEMO")
    print("="*50)