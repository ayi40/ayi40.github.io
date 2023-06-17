---
title: tutorial of PyG
date: 2023/5/23 20:46:25
categories:
  - [GNN,Frame]
---

Some simple knowledge of PyG

<!-- more -->

# Basic

## Data

A single graph in PyG is described by an instance of ```torch_geometric.data.Data```, which holds the following attributes by default:

1. ```data.x```: Node feature matrix ```[num_nodes,num_node_features_dim]```

2. `data.edge_index`: Graph connectivity in [COO format](https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs) with shape `[2, num_edges]` and type `torch.long`

3. `data.edge_attr`: Edge feature matrix with shape `[num_edges, num_edge_features_dim]`

4. `data.y`: Target to train(label). *e.g.*, node-level targets of shape `[num_nodes, *]` or graph-level targets of shape `[1, *]`

   ...

example:

![image-20230523172031612](https://ayimd-pic.oss-cn-guangzhou.aliyuncs.com/image-20230523172031612.png)

```Python
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

##t()会让原变量和新变量直接有依赖（类似浅拷贝），contiguous()断开依赖
data = Data(x=x, edge_index=edge_index.t().contiguous())
>>> Data(edge_index=[2, 4], x=[3, 1])
```

**Note**： Although the graph has only two edges, we need to define four index tuples to account for both directions of an edge.

operation

```python
print(data.keys)
>>> ['x', 'edge_index']

print(data['x'])
>>> tensor([[-1.0],
            [0.0],
            [1.0]])

for key, item in data:
    print(f'{key} found in data')
>>> x found in data
>>> edge_index found in data

'edge_attr' in data
>>> False

data.num_nodes
>>> 3

data.num_edges
>>> 4

data.num_node_features
>>> 1

data.has_isolated_nodes()
>>> False

data.has_self_loops()
>>> False

data.is_directed()
>>> False

# Transfer data object to GPU.
device = torch.device('cuda')
data = data.to(device)
```

## Minibatch

```python
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    batch
    >>> DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    batch.num_graphs
    >>> 32
```



# Message Passing Network

- [(149条消息) pytorch geometric教程一: 消息传递源码详解（MESSAGE PASSING）+实例_每天都想躺平的大喵的博客-CSDN博客](https://blog.csdn.net/weixin_39925939/article/details/121360884)
