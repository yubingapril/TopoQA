import pandas as pd
import numpy as np
import os
from .utils import get_all_col,inter_chain_dis,get_element_index_dis_atom
from torch_geometric import data as DATA
import torch
from joblib import Parallel,delayed
from Bio import PDB


    



def create_graph(model_name,node_dir,vertice_dir,List_cutoff,graph_dir,pdb_dir):
    node_file = os.path.join(node_dir,model_name+'.csv')
    vertice_file = os.path.join(vertice_dir,model_name+'.txt')
    pdb_file = os.path.join(pdb_dir,model_name+'.pdb')   
    try:
        fea_df = pd.read_csv(node_file)
        # label_df=pd.read_csv(label_file)
        vertice_df = pd.read_csv(vertice_file,sep=' ',names=['ID','co_1','co_2','co_3'])
        vertice_df_filter = pd.merge(fea_df,vertice_df,how='left',on='ID')[['ID','co_1','co_2','co_3']]
        fea_col = get_all_col()
        
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein",pdb_file)
        model=structure[0]

        for i in range(len(List_cutoff)):
            curr_cutoff = List_cutoff[i].split("-")
            dis,dis_real=inter_chain_dis.Calculate_distance(vertice_df_filter,curr_cutoff)
            edge_index,edge_attr=get_element_index_dis_atom(dis_real,dis,1.0,vertice_df_filter,model)

            fea = fea_df[fea_col].values

            GCNData =DATA.Data(x=torch.tensor(fea,dtype=torch.float32),
                                        edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                        edge_attr=torch.tensor(edge_attr,dtype=torch.float32))
            # GCNData.__setitem__('model_name', [dataname+'&'+model_name])
            graph_path = os.path.join(graph_dir,model_name+'.pt')
            torch.save(GCNData,graph_path)
    except Exception as e:
        print(f'error in protein {model_name}: {e}')
