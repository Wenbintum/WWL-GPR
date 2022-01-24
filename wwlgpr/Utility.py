import numpy as np
import ray
import ot
import numpy as np
from sklearn.impute import SimpleImputer
import wwlgpr
import os, sys


#! cut graph fragment 
@ray.remote
def multiprocessing_WD(graph_pair_indexs, label_sequences, node_weights):
    return_list = np.array([])
    for graph_pair_index in graph_pair_indexs:
        label1 = graph_pair_index[0]
        label2 = graph_pair_index[1]
        weights_1 = node_weights[label1]
        weights_2 = node_weights[label2+label1]
        # remove node attributes when its weight equals to zero
        filter_ls_1 = label_sequences[label1][np.where(weights_1.flatten() != 0)]
        filter_ls_2 = label_sequences[label2+label1][np.where(weights_2.flatten() != 0)]
        costs = ot.dist(filter_ls_1, filter_ls_2, metric='euclidean')
        W_D_i = ot.emd2(weights_1[weights_1!=0], weights_2[weights_2!=0], costs)
        return_list = np.append(return_list, W_D_i)
    return return_list

def cal_node_weights(graph, atoms, cutoff, inner_cutoff, inner_weight, outer_weight):
    bonding_indexs, ads_bond_indexs = ActiveSiteIndex(graph,atoms)
    total_bonding_index = list(set(bonding_indexs)) + ads_bond_indexs
    assert inner_cutoff <= cutoff
    shortest_path = []
    for i, bonding_index in enumerate(total_bonding_index):
        shortest_path.append(list(map(lambda x:len(x), graph.get_shortest_paths(bonding_index))))
    if len(shortest_path) !=1:
        min_shortest_path = np.amin(shortest_path, axis=0)
    elif len(shortest_path) == 1:
        min_shortest_path = np.array(shortest_path).flatten()
    else:
        raise ValueError('check out shortest_path')
    weights_list = np.zeros(len(atoms))
    weights_list[np.where(min_shortest_path <= inner_cutoff)] = inner_weight
    weights_list[np.where((min_shortest_path > inner_cutoff) & (min_shortest_path <= cutoff))] = outer_weight
    normal_weights_list = weights_list / np.sum(weights_list)
    return normal_weights_list.reshape(-1,1)

def ActiveSiteIndex(graph, atoms):
    ads_bond_index = []; metal_bond_index = [];ensemble_bond_index = []
    for node_i in graph.vs.indices:
        if atoms[node_i].number < 18:
            for edge_i in graph.es[graph.incident(node_i)]:
                if  atoms[edge_i.source].number > 18 and edge_i.source != node_i:
                    metal_bond_index.append(edge_i.source)
                    ads_bond_index.append(node_i)
                if  atoms[edge_i.target].number > 18 and edge_i.target != node_i: 
                    metal_bond_index.append(edge_i.target)
                    ads_bond_index.append(node_i)
    ads_unique_indexs = list(set(ads_bond_index))
    for atom_id in ads_unique_indexs:
        neighbor_index = graph.neighbors(atom_id)
        for neighbor_i in neighbor_index:
            if atoms[neighbor_i].number > 18:
                ensemble_bond_index.append(neighbor_i)
    assert len(ensemble_bond_index) > 0, ('at least ones bonding atom')
    return ensemble_bond_index, ads_unique_indexs

def ConcAttributes(node_attributes):
    conc_attributes = np.asarray(np.concatenate(tuple(i for i in node_attributes),axis=0), dtype=float)                           
    conc_attributes = conc_attributes[:, ~np.all(np.isnan(conc_attributes), axis=0)]
    conc_attributes = conc_attributes[:, ~np.all(conc_attributes == 0, axis=0)]
    return conc_attributes

def CumsumAttributes(conc_attributes, indexlist):
    cumsum_attributes = np.array(np.array_split(conc_attributes, indexlist[:-1]), dtype=object)
    return cumsum_attributes

def HstackAttribute(attributes1, attributes2):
    assert len(attributes1) == len(attributes2)
    conc_attributes1 = np.concatenate(tuple(i for i in attributes1),axis=0)
    conc_attributes2 = np.concatenate(tuple(i for i in attributes2),axis=0)
    hstack_attributes = np.hstack((conc_attributes1, conc_attributes2))
    indexlist = np.cumsum(list(map(lambda node_attribute: len(node_attribute), attributes1)))
    hstack_attributes = np.array_split(hstack_attributes, indexlist[:-1])
    return hstack_attributes

def fill_nan(feature_matrix):
    #return SimpleImputer(missing_values=np.nan,strategy='mean').fit_transform(feature_matrix)
    return SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0.0).fit_transform(feature_matrix)

def OrigianAttribute(node_attributes):
    conc_attributes = np.asarray(np.concatenate(tuple(i for i in node_attributes),axis=0), dtype=float)                             
    conc_attributes = conc_attributes[:, ~np.all(np.isnan(conc_attributes), axis=0)]
    conc_attributes = conc_attributes[:, ~np.all(conc_attributes == 0, axis=0)]
    filled_conc_attributes = fill_nan(conc_attributes)
    indexlist = np.cumsum(list(map(lambda node_attribute: len(node_attribute), node_attributes)))
    origin_node_attributes = np.array_split(filled_conc_attributes, indexlist[:-1])
    return np.array(origin_node_attributes, dtype=object)

def DropFeatures(node_attributes, drop_list):
    conc_attributes = np.asarray(np.concatenate(tuple(i for i in node_attributes),axis=0), dtype=float)                            
    conc_attributes = np.delete(conc_attributes,drop_list-1,axis=1)
    conc_attributes = conc_attributes[:, ~np.all(np.isnan(conc_attributes), axis=0)]
    conc_attributes = conc_attributes[:, ~np.all(conc_attributes == 0, axis=0)]
    indexlist       = np.cumsum(list(map(lambda node_attribute: len(node_attribute), node_attributes)))
    node_attributes = np.array_split(conc_attributes, indexlist[:-1])
    return np.array(node_attributes, dtype=object)

def SplitTrainTest(filenames, species):
    adsorbate_list = []
    for filename in filenames:
        adsorbate  = filename.split('_')[0]
        if adsorbate not in adsorbate_list:
            adsorbate_list.append(adsorbate)         
    ads_dict       = {k:[] for k in adsorbate_list}
    for ads in adsorbate_list:
        for filename in filenames:
            if filename.split('_')[0] == ads:
                ads_dict[ads].append(filename)
    ads_index_list = np.where(np.isin(filenames, ads_dict[species]))
    return ads_index_list

def ClassifySpecies(filenames):
    adsorbate_list = []
    for filename in filenames:
        adsorbate  = filename.split('_')[0]
        if adsorbate not in adsorbate_list:
            adsorbate_list.append(adsorbate)
    classify_list = []
    for filename in filenames:
         key = filename.split('_')[0]
         classify_list.append(np.argwhere(np.array(adsorbate_list) == key)[0][0])
    return classify_list

def writetofile(name,i,j,k):
    for i,j,k in zip(i,j,k):
        with open(name, "a") as infile:
            infile.write(f"{i}, {j}, {k}")
            infile.write("\n")

