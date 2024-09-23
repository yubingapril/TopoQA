import torch
import torchmetrics
from torch import optim
import torch_geometric
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import pandas as pd
import torch.nn as nn
from torch_geometric.data import DataLoader
import numpy as np
import wandb
import random
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from torch_scatter import scatter_mean
from sklearn.metrics import roc_auc_score
import torchsort
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


        # self.embed=torch.nn.Linear(num_feature_xd,hidden_dim) #加入嵌入层
        ##加多层线性层
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
      

    # def configure_optimizers(self):
    #     if self.opt == 'adam':
    #         print('USING ADAM')
    #         optimizer = optim.Adam(self.parameters(),
    #                                lr=self.init_lr)
    #     elif self.opt == 'adamw':
    #         optimizer = optim.AdamW(self.parameters(),
    #                                 lr=self.init_lr)
    #     else:
    #         optimizer = optim.SGD(self.parameters(),
    #                               lr=self.init_lr)
    #     return optimizer

  
    # def training_step(self, train_batch):
    #     # print(len(train_batch))
    #     batch_targets = train_batch[0].y
    #     # print(batch_targets)
    #     # print(train_batch[0].model_name)
    #     batch_scores = self.forward(train_batch)
    #     batch_targets = batch_targets.unsqueeze(1)
    #     # print(batch_scores)
    #     # print(batch_targets)
        
    #     train_mse = self.criterion(batch_scores, batch_targets)

    #     # preds = batch_scores.squeeze()
    #     # targets = batch_targets.squeeze()
    #     # preds = preds.detach().cpu().numpy()
    #     # targets = targets.detach().cpu().numpy()
    #     # spearman_corr, _ = stats.spearmanr(preds, targets)
    #     # spearman_loss = torch.tensor(1 - spearman_corr, requires_grad=True, device=self.device)


    #     # train_loss = train_mse
    #     # print(train_mse)
    #     self.log('train_mse', train_mse, on_step=False, on_epoch=True, sync_dist=True,batch_size=16)
    #     # train_loss=0.5*train_mse+0.5*spearman_loss
    #     # self.log('train_loss',train_loss, on_step=False, on_epoch=True, sync_dist=True)
    #     # return train_mse
    #     return train_mse
    
    # def validation_step(self, val_batch):
    #     batch_targets = val_batch[0].y
    #     batch_scores = self.forward(val_batch)
    #     batch_targets = batch_targets.unsqueeze(1)

    #     val_mse = self.criterion(batch_scores, batch_targets)
    #     val_loss = val_mse 
    #     self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True,batch_size=16)
    #     if 'scores' not in self.validation_step_outputs:
    #         self.validation_step_outputs['scores'] = []
    #     self.validation_step_outputs['scores'].append(batch_scores)
    #     if 'true_scores' not in self.validation_step_outputs:
    #         self.validation_step_outputs['true_scores'] = []
    #     self.validation_step_outputs['true_scores'].append(batch_targets)        
    #     # return {'scores': batch_scores, 'true_score':batch_targets}
    # def on_validation_epoch_end(self):
    #     scores = torch.cat([x for x in self.validation_step_outputs['scores']],dim=0)
    #     true_scores = torch.cat([x for x in self.validation_step_outputs['true_scores']],dim=0)
    #     scores=scores.view(-1).cpu().data.numpy();true_scores=true_scores.view(-1).cpu().data.numpy()
    #     correlation = np.corrcoef(scores, true_scores)[0, 1]
    #     spearman_corr,_=stats.spearmanr(scores, true_scores)
    #     self.log('val_pearson_corr',correlation)
    #     self.log('val_spearman_corr',spearman_corr)
    #     self.validation_step_outputs.clear()
    # def test_step(self, test_batch):
    #     batch_targets = test_batch[0].y
    #     batch_name=test_batch[0].model_name
    #     batch_scores = self.forward(test_batch)
    #     if 'scores' not in self.test_step_outputs:
    #         self.test_step_outputs['scores'] = []
    #     self.test_step_outputs['scores'].append(batch_scores)
    #     if 'true_scores' not in self.test_step_outputs:
    #         self.test_step_outputs['true_scores'] = []
    #     self.test_step_outputs['true_scores'].append(batch_targets)  
    #     if 'name' not in self.test_step_outputs:
    #         self.test_step_outputs['name'] = []
    #     self.test_step_outputs['name'].append(batch_name)  
    # # def test_epoch_end(self,outputs):
    # def on_test_epoch_end(self):
    #     scores = torch.cat([x for x in self.test_step_outputs['scores']],dim=0)
    #     true_scores = torch.cat([x for x in self.test_step_outputs['true_scores']],dim=0)
    #     scores=scores.view(-1).cpu().data.numpy();true_scores=true_scores.view(-1).cpu().data.numpy()
    #     # print(self.test_step_outputs['name'])
    #     test_model_list = [item[0].split('&')[1] for output in self.test_step_outputs['name'] for item in output]
    #     # print(self.test_step_outputs['name'])
    #     # print(test_model_list)
    #     # test_model_list = [item.split('&')[1] for output in self.test_step_outputs['name'] for item in output]
    #     data_name=self.test_step_outputs['name'][0][0][0].split('&')[0].upper()
    #     result_df=pd.DataFrame({'MODEL':test_model_list,'DockQ_wave':true_scores,\
    #                             'pred_dockq_wave':scores})
        
    #     # result_df.to_csv('./table/bm55_999.csv')
    #     # result_df.to_csv('./table/haf2_999.csv')
        
    #     # result_df.to_csv('./haf2.csv')
    #     # result_df.to_csv('./casp15.csv')

    #     dockq_losses=[]
    #     pearson_corrs=[]
    #     spearman_corrs=[]
    #     pdb_list=list(set([x.split('_')[0] for x in result_df['MODEL']]))
    #     for i in pdb_list:
    #         ###result only with pdb i
    #         mask=result_df['MODEL'].str.startswith(i)
    #         curr_df=result_df[mask]

    #         ###cal dockq loss
    #         max_dockq=curr_df['DockQ_wave'].max()
    #         max_index = curr_df['pred_dockq_wave'].idxmax() ###打分模型top1对应的行
    #         model_dockq=curr_df.loc[max_index]['DockQ_wave']
    #         dockq_loss=max_dockq-model_dockq
    #         dockq_losses.append(dockq_loss)
    #         curr_pred=np.array(curr_df['pred_dockq_wave'])
    #         curr_true=np.array(curr_df['DockQ_wave'])
    #         pearson_corr=np.corrcoef(curr_pred,curr_true)[0,1]
    #         spearman_corr,_=stats.spearmanr(curr_pred, curr_true)
    #         pearson_corrs.append(pearson_corr);spearman_corrs.append(spearman_corr)
    #     ####cal correlation coefficient        
    #     dockq_pred=np.array(result_df['pred_dockq_wave'])
    #     dockq_true=np.array(result_df['DockQ_wave'])
    #     pearson_corr=np.corrcoef(dockq_pred,dockq_true)[0,1]
    #     spearman_corr,_=stats.spearmanr(dockq_pred,dockq_true)

    #     ####cal mse mae 
    #     mse = mean_squared_error(dockq_pred,dockq_true)
    #     mae = mean_absolute_error(dockq_pred,dockq_true)

    #     ####cal mean dockq loss
    #     mean_dockq_loss=np.mean(dockq_losses);std_dockq_loss=np.std(dockq_losses)
    #     mean_pearson_corr=np.mean(pearson_corrs);mean_spearman_corr=np.mean(spearman_corrs)

    #     self.log(data_name+'_peason_correlation', pearson_corr)
    #     self.log(data_name+'_spearman_correlation', spearman_corr)
    #     self.log(data_name+'_mean_dockq_loss',mean_dockq_loss)
    #     self.log(data_name+'_std_dockq_loss',std_dockq_loss)
    #     self.log(data_name+'_mean_pearson_corr',mean_pearson_corr)
    #     self.log(data_name+'_mean_spearman_corr',mean_spearman_corr)
    #     self.log(data_name+'_mse',mse)
    #     self.log(data_name+'_mae',mae)
    #     # data_name='test'
    #     # self.log(data_name+'_peason_correlation', pearson_corr)
    #     # self.log(data_name+'_spearman_correlation', spearman_corr)
    #     # self.log(data_name+'_mean_dockq_loss',mean_dockq_loss)
    #     # self.log(data_name+'_std_dockq_loss',std_dockq_loss)
    #     # self.log(data_name+'_mean_pearson_corr',mean_pearson_corr)
    #     # self.log(data_name+'_mean_spearman_corr',mean_spearman_corr)
    #     # self.log(data_name+'_mse',mse)
    #     # self.log(data_name+'_mae',mae)

    #     self.test_step_outputs.clear()
    #     return pearson_corr