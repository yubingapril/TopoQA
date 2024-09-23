from Bio.PDB import *
from Bio.PDB import PDBParser, NeighborSearch
import numpy as np
import pandas as pd
import gudhi
import re

class topo_fea():
    small = 0.01
    def __init__(self,pdbpath,neighbor_dis,e_set,res_list,Cut=8):
        self.pdbpath=pdbpath
        self.neighbor_dis = neighbor_dis
        self.Cut = Cut
        self.e_set = e_set
        self.res_list=res_list
        self.feature_dim = len(e_set)*(5+15)


################neighborhood points
    def find_nei_points(self,descriptor,structure):
    ####c<F>r<152>R<LEU>
       
       #regular expressions
       c_pattern = r'c<([^>]+)>'
       r_pattern = r'r<([^>]+)>'
       i_pattern = r'i<([^>]+)>'  # i< > is optional

       #Find Matches
       c_match = re.search(c_pattern, descriptor)
       r_match = re.search(r_pattern, descriptor)
       i_match = re.search(i_pattern, descriptor)

       #Extract the matching content. If the matching result is None, set the content to None.
       c_content=c_match.group(1) if c_match else None 
       r_content=int(r_match.group(1)) if r_match else None
       i_content=i_match.group(1) if i_match else ' ' 

       res_id=(' ',r_content,i_content)
       trg_res=structure[0][c_content][res_id]

       ####Find the target atom through trg_res
       atom_names=[atom.get_id() for atom in trg_res.get_atoms()]
       if 'CA' in atom_names:
           trg_atom = trg_res['CA']
       else:
           trg_atom = trg_res[atom_names[0]]

       ###Atoms within a certain neighborhood of trg_atom
       atoms = list(structure.get_atoms())
       ns = NeighborSearch(atoms)
       neighbors = ns.search(trg_atom.coord, self.neighbor_dis)
       nei_list=[]
       ###atom type,coord
       for atom in neighbors:
           nei_list.append([atom.get_id(),atom.get_coord()])
       return nei_list
    
    def write_points(self,nei_list,e):
        if e=='all':
            points=[x[1] for x in nei_list]
        else:
            points=[x[1] for x in nei_list if x[0][0] in e]
        return points



    def feature_h0(self, persis):
        Feature_b0 = np.zeros(5)
        tmpbars = np.array([(int(com[0]), float(com[1][0]), float(com[1][1])) for com in persis],
                            dtype=[('dim', int), ('birth', float), ('death', float)])
        bars = tmpbars[(tmpbars['death'] <= self.Cut) & (tmpbars['dim'] == 0) & (tmpbars['death']-tmpbars['birth']>=self.small)]

        # Calculate features
        if len(bars) > 0:
            lengths = bars['death'] - bars['birth']
            Feature_b0[0] = np.sum(lengths)
            Feature_b0[1] = np.min(lengths)
            Feature_b0[2] = np.max(lengths)
            Feature_b0[3] = np.mean(lengths)
            Feature_b0[4] = np.std(lengths)

        return Feature_b0

    def feature_h1h2(self, persis):
        Feature_b1 = np.zeros(15)
        tmpbars = np.array([(int(com[0]), float(com[1][0]), float(com[1][1])) for com in persis],
                            dtype=[('dim', int), ('birth', float), ('death', float)])
        bars = tmpbars[tmpbars['death'] - tmpbars['birth'] >= self.small]

        # Calculate betti1 features
        betti1_bars = bars[bars['dim'] == 1]
        if len(betti1_bars) > 0:
            lengths = betti1_bars['death'] - betti1_bars['birth']
            Feature_b1[0] = np.sum(lengths)
            Feature_b1[1] = np.min(lengths)
            Feature_b1[2] = np.max(lengths)
            Feature_b1[3] = np.mean(lengths)
            Feature_b1[4] = np.std(lengths)
            Feature_b1[5] = np.sum(betti1_bars['birth'])
            Feature_b1[6] = np.min(betti1_bars['birth'])
            Feature_b1[7] = np.max(betti1_bars['birth'])
            Feature_b1[8] = np.mean(betti1_bars['birth'])
            Feature_b1[9] = np.std(betti1_bars['birth'])
            Feature_b1[10] = np.sum(betti1_bars['death'])
            Feature_b1[11] = np.min(betti1_bars['death'])
            Feature_b1[12] = np.max(betti1_bars['death'])
            Feature_b1[13] = np.mean(betti1_bars['death'])
            Feature_b1[14] = np.std(betti1_bars['death'])      
        return Feature_b1

    def cal_fea(self):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', self.pdbpath)
        fea_sum=[]
        for descriptor in self.res_list:
            feature_0d=[];feature_1d=[]
            nei_list=self.find_nei_points(descriptor,structure)
            for element in self.e_set:
                pts=self.write_points(nei_list,element)
                
                ####construct VR complex
                rips_complex = gudhi.RipsComplex(points=pts)
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=1) 
                vr_persis=simplex_tree.persistence()
                f_0=self.feature_h0(vr_persis)
                feature_0d=feature_0d+list(f_0)

                ####construct alpha complex
                ac = gudhi.AlphaComplex(points=pts)
                stree = ac.create_simplex_tree()
                alpha_persis=stree.persistence()
                f_1=self.feature_h1h2(alpha_persis)
                feature_1d=feature_1d+list(f_1)
            fea_sum.append(feature_0d+feature_1d)
        e_set_str=[''.join(element) if isinstance(element, list) else element for element in self.e_set]
        fea_col0=[f'{obj}_{stat}' for obj in ['death'] for stat in ['sum','min','max','mean','std']]
        col_0=[f'f0_{element}_{fea}' for element in e_set_str for fea in fea_col0]

        fea_col1=[f'{obj}_{stat}' for obj in ['len','birth','death'] for stat in ['sum','min','max','mean','std']]
        col_1=[f'f1_{element}_{fea}' for element in e_set_str for fea in fea_col1]
            
        fea_df=pd.DataFrame(fea_sum,columns=col_0+col_1)
        fea_df.insert(0,'ID',self.res_list)

        return fea_df

    



	
