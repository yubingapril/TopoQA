import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re

def get_topo_col():
    e_set=[['C'], ['N'], ['O'], ['C', 'N'], ['C', 'O'], ['N', 'O'], ['C', 'N', 'O']]
    e_set_str=[''.join(element) if isinstance(element, list) else element for element in e_set]
    fea_col0=[f'{obj}_{stat}' for obj in ['death'] for stat in ['sum','min','max','mean','std']]
    col_0=[f'f0_{element}_{fea}' for element in e_set_str for fea in fea_col0]
    fea_col1=[f'{obj}_{stat}' for obj in ['len','birth','death'] for stat in ['sum','min','max','mean','std']]
    col_1=[f'f1_{element}_{fea}' for element in e_set_str for fea in fea_col1]
    topo_col=col_0+col_1
    return topo_col

def get_all_col():
    basic_col=['rasa', 'phi', 'psi', 'SS8_0', 'SS8_1', 'SS8_2', 'SS8_3', 'SS8_4', 'SS8_5', 'SS8_6', 'SS8_7', 'AA_0', 'AA_1', 'AA_2', 'AA_3', 'AA_4', 'AA_5', 'AA_6', 'AA_7', 'AA_8', 'AA_9', 'AA_10', 'AA_11', 'AA_12', 'AA_13', 'AA_14', 'AA_15', 'AA_16', 'AA_17', 'AA_18', 'AA_19', 'AA_20']
    topo_col=get_topo_col()
    col = basic_col + topo_col
    return col


class inter_chain_dis(object):
    def Calculate_distance(Coor_df,arr_cutoff):
        Num_atoms = len(Coor_df)
        Distance_matrix_real = np.zeros((Num_atoms,Num_atoms),dtype=float) 
        Distance_matrix = np.ones((Num_atoms,Num_atoms),dtype=float)
        chain_list=list(Coor_df['ID'].str[2])
        for i in range(Num_atoms):
            for j in range(i,Num_atoms):
                if chain_list[i] == chain_list[j]:
                    Distance_matrix[i][j] = 0.0
                    Distance_matrix[j][i] = 0.0
                    continue
                x_i = float(Coor_df['co_1'][i])
                y_i = float(Coor_df['co_2'][i])
                z_i = float(Coor_df['co_3'][i])

                
                x_j = float(Coor_df['co_1'][j])
                y_j = float(Coor_df['co_2'][j])
                z_j = float(Coor_df['co_3'][j])  
                dis = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)
                if dis <= float(arr_cutoff[0]) or dis >= float(arr_cutoff[1]):
                    Distance_matrix[i][j] = 0.0
                    Distance_matrix[j][i] = 0.0
                else:
                    Distance_matrix[i][j] = 1.0
                    Distance_matrix[j][i] = 1.0
                    Distance_matrix_real[i][j] = dis
                    Distance_matrix_real[j][i] = dis

        return Distance_matrix,Distance_matrix_real
    



def get_pointcloud_type(descriptor1,descriptor2,model,e1,e2):
    ###e1 and e2 can be C N O or all etc.
    #使用正则式表达
    c_pattern = r'c<([^>]+)>'
    r_pattern = r'r<([^>]+)>'
    i_pattern = r'i<([^>]+)>'  # i< > 是可选项

    #查找匹配
    c_match1 = re.search(c_pattern, descriptor1)
    r_match1 = re.search(r_pattern, descriptor1)
    i_match1 = re.search(i_pattern, descriptor1)
    c_match2 = re.search(c_pattern, descriptor2)
    r_match2 = re.search(r_pattern, descriptor2)
    i_match2 = re.search(i_pattern, descriptor2)

    #提取匹配的内容，如果匹配结果为None，则设置内容为None
    c_content1=c_match1.group(1) if c_match1 else None 
    r_content1=int(r_match1.group(1)) if r_match1 else None
    i_content1=i_match1.group(1) if i_match1 else ' ' 
    c_content2=c_match2.group(1) if c_match2 else None 
    r_content2=int(r_match2.group(1)) if r_match2 else None
    i_content2=i_match2.group(1) if i_match2 else ' ' 

    res_id1=(' ',r_content1,i_content1)
    res1=model[c_content1][res_id1]
    res_id2=(' ',r_content2,i_content2)
    res2=model[c_content2][res_id2]

    ####atom coord
    if e1=='all':
        atom_coords1 = [[float(atom.get_coord()[0]),float(atom.get_coord()[1]),
                                   float(atom.get_coord()[1])] for atom in res1.get_atoms()]
    else:
        atom_coords1 = [[float(atom.get_coord()[0]),float(atom.get_coord()[1]),
                                   float(atom.get_coord()[1])] for atom in res1.get_atoms() if atom.get_name()[0]==e1]
    atom_coords1 = np.array(atom_coords1)
    if e2=='all':
        atom_coords2 = [[float(atom.get_coord()[0]),float(atom.get_coord()[1]),
                                   float(atom.get_coord()[1])] for atom in res2.get_atoms()]
    else:    
        atom_coords2 = [[float(atom.get_coord()[0]),float(atom.get_coord()[1]),
                                   float(atom.get_coord()[1])] for atom in res2.get_atoms() if atom.get_name()[0]==e2]
    atom_coords2 = np.array(atom_coords2)
    # print(atom_coords1)
    # print(atom_coords2)
    return atom_coords1,atom_coords2

def distance_of_two_points(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))


def get_dis_histogram(descriptor1,descriptor2,model,e1='all',e2='all'):
    point_cloud1,point_cloud2 = get_pointcloud_type(descriptor1,descriptor2,model,e1,e2)
    number_1=len(point_cloud1);number_2=len(point_cloud2)

    dis_list = sorted([distance_of_two_points(point_cloud1[ind_1], point_cloud2[ind_2]) for ind_1 in range(number_1) for ind_2 in range(number_2)])
    dis_list = np.array(dis_list)

    ##定义区间边界
    bins = np.arange(1,11,1)
    bins=np.append(bins,np.inf) ##添加一个无穷大区间用于包含大于10的值

    ##统计各区间的数量
    hist,_=np.histogram(dis_list,bins=bins)
    return hist



def get_atom_dis(vertice_df,model,edge):
    hist=get_dis_histogram(vertice_df['ID'][edge[0]],vertice_df['ID'][edge[1]],model)

    return hist.tolist()

def get_element_index_dis_atom(mat_re,mat,num,vertice_df_filter,model):
    arr_index = []
    edge_atrr=[]
    
    for i in range(len(mat)):
        for j in range(i+1,len(mat[i])):
            if float(mat[i][j]) == num:
                ###全原子距离
                hists=get_atom_dis(vertice_df_filter,model,[i,j])
                edge_atrr.append([mat_re[i][j]]+hists)
                edge_atrr.append([mat_re[i][j]]+hists)

                arr_index.append([i,j])
                arr_index.append([j,i])
    
    # 将 edge_atrr 转换为 numpy 数组
    edge_atrr = np.array(edge_atrr)
    # 标准化 edge_atrr
    scaler = MinMaxScaler()
    edge_atrr = scaler.fit_transform(edge_atrr)
    return arr_index,edge_atrr