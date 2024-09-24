import torch
import torchmetrics
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn as nn
import pytorch_lightning as pl
from .gat_with_edge import GATv2ConvWithEdgeEmbedding1



class GNN_edge1_edgepooling(pl.LightningModule):
    def __init__(self,pooling_type,num_net=5,hidden_dim=32,edge_dim=1,output_dim=64,n_output=1,heads=8):
        super().__init__()
        self.pooling_type=pooling_type
        self.opt = "adam"
        self.weight_decay=0.002
        self.criterion=torchmetrics.MeanSquaredError()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.heads = heads

        self.num_net=num_net
        num_feature_xd=172



        self.edge_embed=nn.ModuleList([torch.nn.Linear(edge_dim,hidden_dim) for _ in range(self.num_net)])
        self.embed=nn.ModuleList([torch.nn.Linear(num_feature_xd,hidden_dim) for _ in range(self.num_net)])
        self.conv1=nn.ModuleList([GATv2ConvWithEdgeEmbedding1(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=0.25,concat=False).jittable() \
                                  for _ in range(self.num_net)])
        self.conv2=nn.ModuleList([GATv2ConvWithEdgeEmbedding1(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=0.25,concat=False).jittable() \
                                  for _ in range(self.num_net)])       
        self.conv3=nn.ModuleList([GATv2ConvWithEdgeEmbedding1(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=0.25,concat=False).jittable() \
                                  for _ in range(self.num_net)])
        
        self.fc_edge = nn.Linear(hidden_dim,hidden_dim//2)
        self.protein_fc_1=nn.ModuleList([nn.Linear(hidden_dim+hidden_dim//2,output_dim) for _ in range(num_net)])
        
        # combined layers
        
        self.fc1 = nn.Linear(output_dim,64)
        self.out = nn.Linear(64,n_output)
        
        self.validation_step_outputs = {}
        self.test_step_outputs = {}

    def forward(self, data_data):

        for i,module in enumerate(zip(self.embed,self.conv1,self.conv2,self.conv3,self.protein_fc_1,self.edge_embed)):
            data11=data_data[i]
            x,edge_index,edge_attr,batch = data11.x,data11.edge_index,data11.edge_attr,data11.batch
            
            protein_embed=module[0]
            protein_gat1=module[1]
            protein_gat2=module[2]
            protein_gat3=module[3]
            protein_fc1=module[4]
            edge_embed=module[5]
            
        
            x=protein_embed(x)
            edge_attr = edge_embed(edge_attr)
            x,edge_attr=protein_gat1(x,edge_index,edge_attr)
            x=torch.nn.functional.elu(x)
            edge_attr=torch.nn.functional.elu(edge_attr)
            x,edge_attr=protein_gat2(x,edge_index,edge_attr)
            x=torch.nn.functional.elu(x)
            edge_attr=torch.nn.functional.elu(edge_attr)

            
            
            ###获取边的batch索引
            edge_batch_index = batch[edge_index[0]]

            # 池化
            if self.pooling_type == 'add':
                x = global_add_pool(x,batch)
                edge_attr = global_add_pool(edge_attr,edge_batch_index)
            elif self.pooling_type == 'mean':
                x = global_mean_pool(x,batch)
                edge_attr = global_mean_pool(edge_attr,edge_batch_index)

            elif self.pooling_type == 'max':
                x = global_max_pool(x,batch)
                edge_attr = global_max_pool(edge_attr,edge_batch_index)
                # x = torch.sigmoid(self.lin1(x))
                return x,edge_attr
            
            # x = x.to(protein_fc1.weight.device)
            edge_attr = self.fc_edge(edge_attr)
            x_edge = torch.cat((x,edge_attr),dim=1)
            x_edge=protein_fc1(x_edge)
            x_edge=self.relu(x_edge)

            if i == 0:
                x11 = x_edge*1/self.num_net
            else:
                x11 = x11 + x_edge*1/self.num_net
        
        xc=self.fc1(x11)
        xc=self.relu(xc)

        out=self.out(xc)
        out=self.sigmoid(out)

        return out            
      

