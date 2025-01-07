import os 
from src.get_interface import interface_batch
from src.topo_feature import topo_fea
from src.node_fea_df import node_fea
from src.graph import create_graph
from src.proteingat import GNN_edge1_edgepooling
from argparse import ArgumentParser
import pandas as pd 
from joblib import Parallel,delayed
import numpy as np 
import torch
from torch_geometric.data import Data,DataLoader
from torch.utils.data import Dataset
from pathlib import Path


BATCH_SIZE=16

parser = ArgumentParser(description='Evaluate protein complex structures')
parser.add_argument('--complex_folder', '-c', type=str, required=True)
parser.add_argument('--work_dir', '-w', type=str, help='working director to save temporary files', required=True)
parser.add_argument('--result_folder', '-r', type=str, help='The ranking result', required=True)
parser.add_argument('--delete_tmp', '-s', type=bool, help='Save working director or not', default=False, required=False)
args = parser.parse_args()

complex_folder = args.complex_folder
work_dir = args.work_dir
result_folder = args.result_folder
delete_tmp = args.delete_tmp

if not os.path.isdir(complex_folder):
    raise FileNotFoundError(f'Please check complex folder {complex_folder}')
else:
    complex_folder = os.path.abspath(complex_folder)

if len(os.listdir(complex_folder)) == 0:
    raise ValueError(f'The complex folder is empty.')

if not os.path.isdir(work_dir):
    print(f'Creating work folder')
    os.makedirs(work_dir)

if not os.path.isdir(result_folder):
    print(f'Creating result folder')
    os.makedirs(result_folder)
work_dir = os.path.abspath(work_dir)
result_folder = os.path.abspath(result_folder)


interface_dir = os.path.join(work_dir,'interface_ca')
topo_dir=os.path.join(work_dir,'node_topo')
fea_dir=os.path.join(work_dir,'node_fea')
graph_dir=os.path.join(work_dir,'graph')
os.makedirs(interface_dir, exist_ok=True)
os.makedirs(topo_dir, exist_ok=True)
os.makedirs(fea_dir,exist_ok=True)
os.makedirs(graph_dir,exist_ok=True)

####get interface of protein complex
interface_batch(complex_folder,interface_dir,12)


####get topo fea
model_list_all=[file.split('.')[0] for file in os.listdir(complex_folder)]
model_list=[file.split('.')[0] for file in os.listdir(interface_dir) 
            if os.path.getsize(os.path.join(interface_dir,file))!=0]  #filter complex with no interface
no_interface_list=list(set(model_list_all)-set(model_list))
# print(model_list)



def cal_fea_permodel(model_name,pdb_dir,vertice_dir,topo_dir,nei_dis=8):
    ###prepare
    pdb_path=os.path.join(pdb_dir,model_name+'.pdb')
    vertice_file=os.path.join(vertice_dir,model_name+'.txt')
    vertice_df=pd.read_csv(vertice_file,names=['ID','co_1','co_2','co_3'],sep=' ')
    res_list=list(vertice_df['ID'])

    ###cal topological feature
    e_set=[['C'], ['N'], ['O'], ['C', 'N'], ['C', 'O'], ['N', 'O'], ['C', 'N', 'O']]
    # obj=topo_fea(pdb_path,8,['C','all'],['c<A>r<4>R<ARG>'])
    obj=topo_fea(pdb_path,nei_dis,e_set,res_list)
    df=obj.cal_fea()
    # print(df)
    df_save_path=os.path.join(topo_dir,model_name+'.csv')
    df.to_csv(df_save_path,index=False)

def topo_batch(pdb_dir,vertice_dir,topo_dir,nei_dis=8,n=12):
    # model_list=[file.split('.')[0] for file in os.listdir(pdb_dir)]
    Parallel(n_jobs=n)(
        delayed(cal_fea_permodel)(model,pdb_dir,vertice_dir,topo_dir,nei_dis) for model in model_list
    )

topo_batch(complex_folder,interface_dir,topo_dir)



####calculate node features
def get_fea(model_name,pdb_dir,topo_dir,vertice_dir,fea_dir):
    try:
        fea_df,_ = node_fea(model_name,pdb_dir,vertice_dir,topo_dir).calculate_fea()
        pd.set_option('future.no_silent_downcasting', True)
        fea_df.replace('NA', np.nan, inplace=True)
        fea_df_clean=fea_df.dropna()
        fea_df_path = os.path.join(fea_dir,model_name+'.csv')
        fea_df_clean.to_csv(fea_df_path,index=False)
    except Exception as e:
        print(f'error in {model_name}: {e}')
Parallel(n_jobs=12)(
    delayed(get_fea)(model,complex_folder,topo_dir,interface_dir,fea_dir) for model in model_list
)


####get graph
arr_cutoff=['0-10']
Parallel(n_jobs=12)(
    delayed(create_graph)(model,fea_dir,interface_dir,arr_cutoff,graph_dir,complex_folder) for model in model_list
)

####predict 
class GraphDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        graph_path = self.file_paths[idx]
        graph_data = torch.load(graph_path)
        return [graph_data]

def collate_fn(batch):
    return Data.from_data_list(batch)

def get_loader(graph_path,model_list):
    graph_list = [os.path.join(graph_path,model+'.pt') for model in model_list]
    dataset = GraphDataset(graph_list)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False,num_workers=4)
    return data_loader

    
model_list=[file.split('.')[0] for file in os.listdir(interface_dir)         
            if os.path.getsize(os.path.join(interface_dir,file))!=0]
eval_loader = get_loader(graph_dir,model_list)
#load trained model
current_path = Path().absolute()
ckpt_file = f'{current_path}/model/topoqa.ckpt'
checkpoint=torch.load(ckpt_file)
model = GNN_edge1_edgepooling('mean',num_net=1,edge_dim=11,heads=8)
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # turn on model eval mode

pred_dockq = []
for idx, batch_graphs in enumerate(eval_loader):
    batch_scores = model.forward(batch_graphs)
    pred_dockq.extend(batch_scores.cpu().data.numpy().tolist())
pred_dockq = [i[0] for i in pred_dockq]
df = pd.DataFrame(list(zip(model_list, pred_dockq)), columns=['MODEL', 'PRED_DOCKQ'])
df.to_csv(os.path.join(result_folder, 'result.csv'), index=False)


