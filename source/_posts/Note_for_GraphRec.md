---
title: GraphRec
date: 2023/2/24 20:46:25
categories:
  - [RecSys, SocialRec]
---

.

<!-- more -->


# GraphRec

# GraphRec feature

1. Can capture both interactions and opinions in user-item graph.

2. Consider different strengths of social relations.

3. Use attention mechanism.

# Overall architecture

![Snipaste_2023-02-02_15-10-34](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/Snipaste_2023-02-02_15-10-34.png)


### Three import module:

1. User Modeling: used to compute User Latent Factor(vector containing many useful information)

2. Item Modeling: used to compute Item Latent Factor.

3. Rating Prediction: used to predict the item which user would like to interact with.


# Source code analyses

## Data

### **What kind of datas we use?**
1. User-Item graph: record interation(e.g. purchase) and opinion(e.g. five star rating) between user and item

2. User-User social graph: relationship between user and user

### **How to represent these datas in code?**

#### User-Item graph:

1. history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)

```python
history_u_list = {
    user_id1:[item_id1, item_id2, item_id3...],
    user_id2:[item_id4...],
}
history_ur_list = {
    user_id1:[rating_score_u1i1, rating_score_u1i2, rating_score_u1i3...],
    user_id2:[rating_score_u2i4...],
}

e.g.
history_u_list = {
    681: [0, 156], 
    81: [1, 41, 90]}
history_ur_list = {
    681: [5,4],
    81: [4,3,2]}
```

2. history_v_lists, history_vr_lists: user set (in training set) who have interacted with the item, and rating score (dict). Similar with history_u_lists, history_ur_lists but key is item id and value is user id.

#### User-User socal graph

1. social_adj_lists: user's connected neighborhoods

```python
social_adj_lists = {
    user_id1:[user_id2, user_id3, user_id4...],
    user_id2:[user_id1...],
}
```

#### other

1.  train_u, train_v, train_r: used for model training, one by one based on index (user, item, rating)

```python
train_u = [user_id1, user_id2,....]
train_v = [item_id34, item_id1,...]
train_r = [rating_socre_u1i34, rating_socre_u2i1]
len(train_u) = len(train_v) = len(train_r)
```
2. test_u, test_v, test_r: similar with training datas

3. ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
```
{2.0: 0, 1.0: 1, 3.0: 2, 4.0: 3, 2.5: 4, 3.5: 5, 1.5: 6, 0.5: 7}
```

### **How to pre-process data?**

use `torch.utils.data.TensorDataset` and `torch.utils.data.DataLoader` generate `training_dataset` and `testing_dataset` (user, item, rating)

```python
support batchsize = 5
[tensor([409,  88, 134, 298, 340]),                             #user id
tensor([1221,  761,   39,  145,    0]),                         #item id
tensor([1.0000, 2.0000, 3.5000, 0.5000, 1.5000, 3.5000])        #rating score
]
```

## Model

### Init

Translate user_id, item_id and rating_id to low-dimension vector, just random initize, the weight of embedding layers will be trained.

After translate we get 

    qj-embedding of item vj, 
    pi-embedding of user ui, 
    er-embedding of rating.

```python
u2e = nn.Embedding(num_users, embed_dim).to(device)
v2e = nn.Embedding(num_items, embed_dim).to(device)
r2e = nn.Embedding(num_ratings, embed_dim).to(device)
print(u2e, v2e, r2e)

'''Output
Embedding(705, 64) 
Embedding(1941, 64) 
Embedding(8, 64)
'''
```

So that, we can easily get embedding through U2e, V2e and r2e.

### Overall architecture

![GraphRec](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/GraphRec.jpg)

GraphRec consist of User Modeling, Item Modeling and Rating Prediction.
The forward code of GraphRec is as follow:

```python
class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        ...

    def forward(self, nodes_u, nodes_v):
        # nodes_u : [128] 128(batchsize) user id
        # nodes_v : [128] 128(batchsize) item id
        # self.enc_u is the User Modeling part(including Item Aggregation and Social Aggregation )
        # self.enc_v_history is the Item Modeling part(User Aggregation)
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        # After aggregation information, forward two layer MLP， and get the Latent vector of user and item
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)
        
        # concatenated user vector and item vector, use three layer MLP to predict
        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()
    
    def loss(self, nodes_u, nodes_v, labels_list):
        ...

```

full code of GraphRec class

<details>

```python
class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)

```

</details>

### User Modeling

It contain Item Aggregation and Social Aggregation

在这里本质上是先做了一层Item Aggregation之后，用得到的结果再做一层Social Aggregation
所以这里的Item Aggregation，本质上是Social Aggregation中的self-connection

```python
class Social_Encoder(nn.Module):

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="cpu"):
        ...

    def forward(self, nodes):

        # to_neighs is a list which element is list recording social neighbor node, and len(list) is batchsize,
        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_adj_lists[int(node)])

        # Social aggregation
        neigh_feats = self.aggregator.forward(nodes, to_neighs)  # user-user network

        # Item aggregation
        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = self_feats.t()
        
        # self-connection could be considered.
        # Concatenate Item Aggregation and Social Aggregation, and through one layer MLP
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
```

full code of User Modeling

<details>

```python
class Social_Encoder(nn.Module):

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="cpu"):
        super(Social_Encoder, self).__init__()

        self.features = features
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):

        # to_neighs is a list which element is list recording social neighbor node, and len(list) is batchsize,
        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_adj_lists[int(node)])

        # Item aggregation
        neigh_feats = self.aggregator.forward(nodes, to_neighs)  # user-user network

        # Social aggregation
        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = self_feats.t()
        
        # self-connection could be considered.
        # Concatenate Item Aggregation and Social Aggregation, and through one layer MLP
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
```

</details>

#### Item Aggregation

```python
class UV_Encoder(nn.Module):

    def __init__(self, features, embed_dim, history_uv_lists, history_r_lists, aggregator, cuda="cpu", uv=True):
        ...

    def forward(self, nodes):
        tmp_history_uv = []
        tmp_history_r = []

        #get nodes(batch) neighbors
        #tmp_history_uv is a list which len is 128,while it's element is also a list meaning that the each node's(in batch) neighbor item id list
        #tmp_history_r is similar with tmp_history_uv, but record the rating score instead of item id
        for node in nodes:
            tmp_history_uv.append(self.history_uv_lists[int(node)])
            tmp_history_r.append(self.history_r_lists[int(node)])

        # after neigh aggregation
        neigh_feats = self.aggregator.forward(nodes, tmp_history_uv, tmp_history_r)  # user-item network

        # id to embedding (features : u2e)
        self_feats = self.features.weight[nodes]
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
```
And the  ```self.aggregator``` in neigh aggregation is:

```python
class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        ...

    def forward(self, nodes, history_uv, history_r):
        # create a container for result, shpe of embed_matrix is (batchsize,embed_dim)
        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        # deal with each single nodes' neighbors
        for i in range(len(history_uv)):
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]

            # e_uv : turn neighbors id to embedding
            # uv_rep : turn single node to embedding
            if self.uv == True:
                # user component
                e_uv = self.v2e.weight[history]
                uv_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_uv = self.u2e.weight[history]
                uv_rep = self.v2e.weight[nodes[i]]

            # get rating score embedding
            e_r = self.r2e.weight[tmp_label]
            # concatenated rating and neighbor, and than through two layers mlp to get xia
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))

            o_history = F.relu(self.w_r2(x))
            # calculate neighbor attention and xia*weight to finish aggregation
            att_w = self.att(o_history, uv_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        # result (batchsize, embed_dim)
        to_feats = embed_matrix
        return to_feats
```

While ```self.att``` is:

```python
class Attention(nn.Module):
    def __init__(self, embedding_dims):
        ...

    def forward(self, node1, u_rep, num_neighs):
        # pi
        uv_reps = u_rep.repeat(num_neighs, 1)
        # concatenated neighbot and pi
        x = torch.cat((node1, uv_reps), 1)
        # through 3 layers MLP
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        # get weights
        att = F.softmax(x, dim=0)
        return att
```

#### Social Aggregation

use the result of Item Aggregation and pi as input

```python
class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, embed_dim, cuda="cpu"):
        ...

    def forward(self, nodes, to_neighs):
        #return a uninitialize matrix as result container, which shape is (batchsize, embed_dim)
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(nodes)):
            # get social graph neighbor
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            
            # fase : can use user embedding instead of result of item aggregation to improve speed
            # e_u = self.u2e.weight[list(tmp_adj)] # fast: user embedding 
            # slow: item-space user latent factor (item aggregation)
            feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            e_u = torch.t(feature_neigbhors)

            u_rep = self.u2e.weight[nodes[i]]
            
            # concatenated node embedding and neigbor vector (result of item aggregation) 
            # and than through MLPs and Softmax to calculate weights
            att_w = self.att(e_u, u_rep, num_neighs)
            # weight*neighbor vector
            att_history = torch.mm(e_u.t(), att_w).t()
            embed_matrix[i] = att_history
        to_feats = embed_matrix

        return to_feats
```

### Item Modeling

Similar with the Item Aggregation of User Modeling

```python
class UV_Encoder(nn.Module):

    def __init__(self, features, embed_dim, history_uv_lists, history_r_lists, aggregator, cuda="cpu", uv=True):
        ...

    def forward(self, nodes):
        tmp_history_uv = []
        tmp_history_r = []

        #get nodes(batch) neighbors of item
        for node in nodes:
            tmp_history_uv.append(self.history_uv_lists[int(node)])
            tmp_history_r.append(self.history_r_lists[int(node)])

        # after neigh aggregation
        neigh_feats = self.aggregator.forward(nodes, tmp_history_uv, tmp_history_r)  # user-item network

        # id to embedding (features : v2e)
        self_feats = self.features.weight[nodes]
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
```
And the  ```self.aggregator``` in neigh aggregation is:

```python
class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        ...

    def forward(self, nodes, history_uv, history_r):
        # create a container for result, shpe of embed_matrix is (batchsize,embed_dim)
        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        # deal with each single item nodes' neighbors
        for i in range(len(history_uv)):
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]

            # e_uv : turn neighbors(user node) id to embedding
            # uv_rep : turn single node(item node) to embedding
            if self.uv == True:
                # user component
                e_uv = self.v2e.weight[history]
                uv_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_uv = self.u2e.weight[history]
                uv_rep = self.v2e.weight[nodes[i]]

            # get rating score embedding
            e_r = self.r2e.weight[tmp_label]
            # concatenated rating and neighbor, and than through two layers mlp to get fjt
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))

            o_history = F.relu(self.w_r2(x))
            # calculate neighbor attention and fjt*weight to finish aggregation
            att_w = self.att(o_history, uv_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        # result (batchsize, embed_dim)
        to_feats = embed_matrix
        return to_feats
```