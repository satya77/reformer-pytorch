from reformer_pytorch.reformer_pytorch import  *

class ReformerWithEmbedding(nn.Module):
    '''
    The Reformer Language model but with the positional embeddings removed.
    '''
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 4, add_local_attn_hash = False, ff_chunks = 100, attn_chunks = 1, causal = False, weight_tie = False, lsh_dropout = 0., ff_dropout = 0., ff_mult = 4, ff_activation = None, post_attn_dropout = 0., layer_dropout = 0., random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False, full_attn_thres = 0, reverse_thres = 0, num_mem_kv = 0, one_value_head = False, emb_dim = None, return_embeddings = False, weight_tie_embedding = False):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.token_emb.weight.data.uniform_(-0.01, 0.01)

        self.to_model_dim = Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        self.reformer = Reformer(dim, depth, max_seq_len, heads = heads, bucket_size = bucket_size, n_hashes = n_hashes, add_local_attn_hash = add_local_attn_hash, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, ff_mult = ff_mult, ff_activation = ff_activation, ff_dropout = ff_dropout, post_attn_dropout = 0., layer_dropout = layer_dropout, random_rotations_per_head = random_rotations_per_head, twin_attention = twin_attention, use_scale_norm = use_scale_norm, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, reverse_thres = reverse_thres, num_mem_kv = num_mem_kv, one_value_head = one_value_head)

        if return_embeddings:
            self.out = Identity()
            return

        self.out = nn.Sequential(
            nn.Linear(dim, emb_dim) if emb_dim != dim else Identity(),
            nn.Linear(emb_dim, num_tokens) if not weight_tie_embedding else MatrixMultiply(self.token_emb.weight, transpose=True, normalize=True)
        )

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = self.to_model_dim(x)
        x = self.reformer(x, **kwargs)
        return self.out(x)



class SimpleClassifier(nn.Module):
    '''
        Simple classifier on top of the reformers
        '''
    def __init__(self,input_size=1024,emb_dim=100,outpu_size=1):
        super().__init__()
        self.fc1=nn.Linear(input_size,emb_dim)
        self.fc2=nn.Linear(emb_dim,outpu_size)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x1,x2):

        x = torch.cat([x1, x2], 1)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class ReformerSiamese(nn.Module):
    def __init__(self,emb_dim,num_tokens=20000,max_seq_len=8192,device="cuda"):
        super().__init__()
        #param
        self.reformer= ReformerWithEmbedding(
                    num_tokens=num_tokens,
                    dim=emb_dim,
                    depth=12,
                    max_seq_len=max_seq_len,
                    heads=8,
                    lsh_dropout=0.1,
                    ff_dropout=0.1,
                    post_attn_dropout=0.1,
                    layer_dropout=0.1,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
                    causal=True,  # auto-regressive or not
                    bucket_size=64,  # average size of qk per bucket, 64 was recommended in paper
                    n_hashes=4,  # 4 is permissible per author, 8 is the best but slower
                    emb_dim=emb_dim,  # embedding factorization for further memory savings
                    ff_chunks=200,  # number of chunks for feedforward layer, make higher if there are memory issues
                    attn_chunks=8,  # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
                    num_mem_kv=128,  # persistent learned memory key values, from all-attention paper
                    twin_attention=False,  # both branches of the reversible network will be attention
                    full_attn_thres=1028,  # use full attention if context length is less than set value
                    reverse_thres=1024,
                    # turn off reversibility for 2x speed for sequence lengths shorter or equal to the designated value
                    use_scale_norm=False,  # use scale norm from 'Transformers without tears' paper
                    one_value_head=False,  # use one set of values for all heads from 'One Write-Head Is All You Need'
                    weight_tie=False,  # tie parameters of each layer for no memory per additional depth
                    weight_tie_embedding=True,  # use token embedding for projection of output, some papers report better results
                    use_full_attn=True
                ).to(device)

        self.classifier=SimpleClassifier(input_size=2*30522,emb_dim=emb_dim).to(device)

    def forward(self,x1,x2):

        embed1= self.reformer(x1)
        embed2 = self.reformer(x2)

        sum_embeddings1 = torch.sum(embed1, 1)/embed1.size(1)
        sum_embeddings2 = torch.sum(embed2, 1)/embed2.size(1)

        x= self.classifier(sum_embeddings1,sum_embeddings2)
        return x
