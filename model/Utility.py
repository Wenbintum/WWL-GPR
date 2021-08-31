from sklearn.metrics import mean_absolute_error, mean_squared_error
from skopt import gp_minimize
from functools import partial
import numpy as np
import ray
import ot
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
import numpy as np
from sklearn.base import TransformerMixin
import igraph as ig
from collections import defaultdict
from typing import List
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from scipy.linalg import cholesky, cho_solve, decomp_schur, solve_triangular


class GraphBase:
    def __init__(self,
                 db_graphs,
                 db_atoms,
                 node_attributes,
                 y,
                 drop_list = None,
                 num_iter  = 1,
                 post_type = "Normalize",
                 filenames = None
                 ):
        """
        Basic setting of graph model
        Args:
                      db_graphs         : graph representations of initial guess generated from igraph or xnetwork
                      db_atoms          : ASE atoms object of initial guess
                      node_attributes   : primary features or feature vectors in the shape of samples x atoms x features
                      y                 : target properties of interest
            drop_list (optional)        : index of features to be removed. Defaults to None.
            num_iter  (int, optional)   : number of Weisfeiler-Lehman refinement steps. Defaults to 1.
            post_type (str, optional)   : preprocessing of data. Defaults to "Standardization".
            filenames (list, optional)  : filenames of all samples in the database. Defaults to None.
        """
        self.db_graphs            = np.asarray(db_graphs)
        self.db_atoms             = np.asarray(db_atoms)
        self.node_attributes      = np.asarray(node_attributes, dtype=object)
        self.y                    = np.asarray(y)
        self.filenames            = np.asarray(filenames)

        self.drop_list = drop_list
        self.num_iter  = num_iter
        self.post_type = post_type
        self.post_node_attributes = self.__class__.Post_input(db_graphs       = self.db_graphs,
                                                              node_attributes = self.node_attributes,
                                                              drop_list       = self.drop_list,
                                                              post_type       = self.post_type,
                                                              num_iter        = self.num_iter)
    @staticmethod
    def gpr_kernel_matrix(db_graphs, 
                          db_atoms, 
                          post_node_attributes, 
                          cutoff, 
                          inner_cutoff, 
                          inner_weight, 
                          outer_weight,
                          gpr_gamma, 
                          gpr_sigma,
                          num_cpus = 40
                          ):
        """
        Calculating the precomputed kernel matrix based on certain hyperparameters.
        This function will be used in the process of hyperparameter optimization of training as well as prediction
        Args:
            post_node_attributes (matrix): node attributes after data preprocessing
            cutoff               (int)   : number of layer to be considered
            inner_cutoff         (int)   : cutoff for inner layers
            inner_weight         (float) : weight for the atoms in inner layers
            outer_weight         (float) : weight for the atoms in outer layers
            gpr_gamma            (float) : hyperparameter of gamma. Defaults to be 1
            gpr_sigma            (float) : hyperparameter of sigma

        Returns:
            M [type]: precomputed kernel matrix
        """
        n = len(db_graphs)
        res_node_weight = np.zeros(n, dtype=object)
        for graph_index, graph in enumerate(db_graphs):
            res_node_weight[graph_index] = cal_node_weights(graph, db_atoms[graph_index], cutoff, inner_cutoff, inner_weight, outer_weight)
        #*2 calculate WD
        post_node_attributes = ray.put(post_node_attributes)   # node features
        node_weights         = ray.put(res_node_weight)        # node weights
        M = np.zeros((n,n))
        triu = np.triu_indices(n)
        graph_pair_index     = np.asarray([[i,j] for i in range(n) for j in range(n-i)])    #pair of index to be calculated
        splited_graph        = np.array_split(graph_pair_index, num_cpus)                   #! number of cpu to be use.
        result_ids = [multiprocessing_WD.remote(ray.put(graph_index), post_node_attributes, node_weights) for graph_index in splited_graph]
        res = np.concatenate(ray.get(result_ids))
        M[triu] = res.ravel()
        M = (M + M.T)
        M = gpr_sigma**2*linear_kernel(M/gpr_gamma)           #linear pairwise kernel to deal with wasserstain distance value.
        return M

    @staticmethod
    def Post_input(db_graphs, node_attributes, drop_list, post_type, num_iter):
        """
        Postprocessing node attributes
        1. features that you may want to remove (optional)
        2. method of data preprocessing, Normalization, Standardization or using Raw data
        3. how many Weisfeiler-Lehman refinement steps. 
        """
        if drop_list is not None:
            node_attributes = DropFeatures(node_attributes, drop_list)
        if post_type == "Normalize":
            post_node_attributes = NormalizeAttribute(node_attributes)
        elif post_type == "Original" :
            post_node_attributes = OrigianAttribute(node_attributes)
        elif post_type == "Standardize":
            post_node_attributes = StandardAttribute(node_attributes)
        else:
            raise TypeError('Unkown postprocessing')
        post_node_attributes     = R_conv_attributes(db_graphs, post_node_attributes, num_iterations=num_iter)
        return post_node_attributes

class BayOptCv(GraphBase):
    """
    Bayesian optimization 
    """
    # default setting of hyperparameters. The dictionary is going to be updated when doing Bayesian optimization
    default_gpr_hyper = {"cutoff"       : None,
                    "inner_cutoff"      : None,
                    "inner_weight"      : None,
                    "outer_weight"      : None,
                    "noise_level"         : 0.1,
                    "gpr_gamma"         : 1,
                    "gpr_sigma"         : 1
                    }
    
    def __init__(self,
                 num_cpus = 40, 
                 classifyspecies = None,
                 *args,
                 **kwgs,
                 ):
        """[summary]

        Args:
            classifyspecies ([type], optional): database is stratified by adsorbates. Defaults to None.
        """
        super().__init__(*args, **kwgs)
        self.num_cpus        = num_cpus
        self.classifyspecies = classifyspecies
        self.train           = False
        
    def _update_default_hypers(self, hyperpars, name_hypers, fix_hypers):
        input_hyper_dict = dict(zip(name_hypers, hyperpars))
        BayOptCv.default_gpr_hyper.update(fix_hypers)
        BayOptCv.default_gpr_hyper.update(input_hyper_dict)

    def _Predict_prepare(self, test_graphs, test_atoms, test_node_attributes):
        """
        prepare test dataset for prediction

        Args:
            test_graphs ([type]): [description]
            test_atoms ([type]): [description]
            test_node_attributes ([type]): 

        Returns:
        conc_db_graphs: concatenated graph representations (training + test)
        conc_db_atoms : concatenated atoms object (training + test)
        """
        assert len(test_node_attributes) > 1
        conc_db_graphs  = np.append(self.db_graphs, test_graphs)
        conc_db_atoms   = np.append(self.db_atoms,  test_atoms)
        conc_attributes = np.concatenate(tuple(i for i in self.node_attributes) + 
                                         tuple(j for j in test_node_attributes), axis=0)
        indexlist = np.cumsum(list(map(lambda node_attribute: len(node_attribute), self.node_attributes)) + 
                    list(map(lambda node_attribute: len(node_attribute), test_node_attributes)))
        conc_attributes = np.array_split(conc_attributes, indexlist[:-1])
        self.post_node_attributes = self.__class__.Post_input(
                                        db_graphs      = conc_db_graphs,
                                        node_attributes= conc_attributes,
                                        drop_list      = self.drop_list,
                                        post_type      = self.post_type,
                                        num_iter       = self.num_iter
                                        )

        return conc_db_graphs, conc_db_atoms
    
    def Predict(self, test_graphs, test_atoms, test_node_attributes, test_target, optimized_hypers, name_hypers):
        """[summary]
        Args:
            test_graphs ([type]): [description]
            test_atoms ([type]): [description]
            test_node_attributes ([type]): [description]
            test_target ([type]): [description]
            optimized_hypers ([type]): optimized hyperparameters taken from training process
            name_hypers ([type]):      hyperparameters to be optimize, others are fixed as default setting

        Returns:
            [type]: [description]
        """
        print("GPR prediction beginning",self.train)
        conc_db_graphs, conc_db_atoms = self._Predict_prepare(test_graphs, test_atoms, test_node_attributes)
        BayOptCv.default_gpr_hyper.update(dict(zip(name_hypers, optimized_hypers)))
        cutoff, inner_cutoff, inner_weight, outer_weight, noise_level, gpr_gamma, gpr_sigma = BayOptCv.default_gpr_hyper.values()
        total_n = len(conc_db_graphs)
        conc_kernel_matrix = self.__class__.gpr_kernel_matrix(db_graphs=conc_db_graphs, db_atoms=conc_db_atoms, 
                                                                post_node_attributes=self.post_node_attributes, 
                                                                cutoff=cutoff, inner_cutoff=inner_cutoff, 
                                                                inner_weight=inner_weight, outer_weight=outer_weight,
                                                                gpr_gamma = gpr_gamma, gpr_sigma=gpr_sigma
                                                                ,num_cpus =self.num_cpus)
        total_diag = np.zeros((total_n, total_n))
        np.fill_diagonal(total_diag, [noise_level]*int(total_n - len(test_target))+[0]*len(test_target))
        conc_kernel_matrix = conc_kernel_matrix + total_diag
        train_matrix  = conc_kernel_matrix[:len(self.y), :len(self.y)] + 1e-8 * np.eye(len(self.y))
        test_matrix   = conc_kernel_matrix[len(self.y):, :len(self.y)]
        Kfy           = conc_kernel_matrix[:len(self.y), len(self.y):]
        Kff_inv       = np.linalg.inv(train_matrix + 1e-8 * np.eye(len(self.y)))
        mu            = Kfy.T.dot(Kff_inv).dot(self.y)
        test_RMSE     = mean_squared_error(mu,  test_target,  squared=False)
        return test_RMSE, mu

    def _LossFunc(self, hyperpars, name_hypers, fix_hypers):
        """
        loss function: minimize likelihood in the case of gaussian process regression
        using fix_hypers to fix some hyperparameters that you don't want to optimize
        Args:
            hyperpars ([type]): [description]
            name_hypers ([type]): [description]
            fix_hypers ([type]): [description]
        """
        #print(self.train, "gpr training")
        self._update_default_hypers(hyperpars, name_hypers, fix_hypers)
        cutoff, inner_cutoff, inner_weight, outer_weight, noise_level, gpr_gamma, gpr_sigma = self.default_gpr_hyper.values()
        pre_kernel = self.__class__.gpr_kernel_matrix(db_graphs=self.db_graphs, db_atoms=self.db_atoms, 
                                                    post_node_attributes=self.post_node_attributes
                                                    ,cutoff=cutoff, inner_cutoff=inner_cutoff, 
                                                    inner_weight=inner_weight, outer_weight=outer_weight,
                                                    gpr_gamma=gpr_gamma, gpr_sigma=gpr_sigma,
                                                    num_cpus=self.num_cpus)
        if self.train == False:
            pre_kernel = pre_kernel + noise_level*np.eye(len(self.y))
        Kxx = pre_kernel
        train_y = self.y[:, np.newaxis] 
        L = cholesky(Kxx, lower=True)
        alpha = cho_solve((L,True), train_y)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", train_y, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= Kxx.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)
        return -log_likelihood

    def BayOpt(self, name_hypers, dimensions, fix_hypers, default_para):
        """
        Bayeisan setting itself. Please check skopt for details
        """
        res = gp_minimize(
                        func            = partial(self._LossFunc, name_hypers=name_hypers, fix_hypers=fix_hypers),
                        dimensions      = dimensions,
                        n_calls         = 50, #
                        n_random_starts = 30,
                        acq_func        = "EI",
                        x0              = default_para,
                        xi              = 0.01,
                        random_state    = 0,
                        )
        self.train = True
        # plot_convergence(res)
        # plt.savefig('convergence.png'); plt.close()
        return res

class ContinuousWeisfeilerLehman(TransformerMixin):
    """
    Class that implements the continuous Weisfeiler-Lehman propagation scheme
    """
    def __init__(self):
        self._results = defaultdict(dict)
        self._label_sequences = []

    def _preprocess_graphs(self, X: List[ig.Graph]):
        """
        Load graphs from gml files.
        """
        # initialize
        node_features = []
        adj_mat = []
        n_nodes = []
        # Iterate across graphs and load initial node features
        for graph in X:
            if not 'label' in graph.vs.attribute_names():
                graph.vs['label'] = list(map(str, [l for l in graph.vs.degree()]))    
            # Get features and adjacency matrix
            node_features_cur = graph.vs['label']
            adj_mat_cur = np.asarray(graph.get_adjacency().data)
            # Load features
            node_features.append(np.asarray(node_features_cur).astype(float).reshape(-1,1))
            adj_mat.append(adj_mat_cur.astype(int))
            n_nodes.append(adj_mat_cur.shape[0])

        # By default, keep degree or label as features, if other features shall
        # to be used (e.g. the one from the TU Dortmund website), 
        # provide them to the fit_transform function.
        n_nodes = np.asarray(n_nodes)
        node_features = np.asarray(node_features, dtype=object)
        return node_features, adj_mat, n_nodes

    def _create_adj_avg(self, adj_cur):
        '''
        create adjacency
        '''
        deg = np.sum(adj_cur, axis = 1)
        deg = np.asarray(deg).reshape(-1)

        deg[deg!=1] -= 1

        deg = 1/deg
        deg_mat = np.diag(deg)
        adj_cur = adj_cur.dot(deg_mat.T).T
        return adj_cur

    def fit_transform(self, X: List[ig.Graph], node_features = None, num_iterations: int=3):
        """
        Transform a list of graphs into their node representations. 
        Node features should be provided as a numpy array.
        """
        node_features_labels, adj_mat, n_nodes = self._preprocess_graphs(X)
        if node_features is None:
            node_features = node_features_labels
        #wenbin
        # node_features_data = scale(np.concatenate(node_features, axis=0), axis = 0)
        # splits_idx = np.cumsum(n_nodes).astype(int)
        # node_features_split = np.vsplit(node_features_data,splits_idx)		
        # node_features = node_features_split[:-1]
        node_features = node_features  #wenbin
        # Generate the label sequences for h iterations
        n_graphs = len(node_features)
        self._label_sequences = []
        for i in range(n_graphs):
            graph_feat = []
            for it in range(num_iterations+1):
                if it == 0:
                    graph_feat.append(node_features[i])
                else:
                    adj_cur = adj_mat[i]+np.identity(adj_mat[i].shape[0])
                    adj_cur = self._create_adj_avg(adj_cur)
                    np.fill_diagonal(adj_cur, 0)
                    graph_feat_cur = 0.5*(np.dot(adj_cur, graph_feat[it-1]) + graph_feat[it-1])
                    # import pdb; pdb.set_trace()
                    graph_feat.append(graph_feat_cur)

            self._label_sequences.append(np.concatenate(graph_feat, axis = 1))
        return self._label_sequences

def fill_nan(feature_matrix):
    #! 0 padding for the features of None
    return SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0.0).fit_transform(feature_matrix)

def NormalizeAttribute(node_attributes):
    conc_attributes = np.asarray(np.concatenate(tuple(i for i in node_attributes),axis=0), dtype=float)                           
    conc_attributes = conc_attributes[:, ~np.all(np.isnan(conc_attributes), axis=0)]
    conc_attributes = conc_attributes[:, ~np.all(conc_attributes == 0, axis=0)]
    normed_conc_attributes = conc_attributes / np.nanmax(np.abs(conc_attributes),axis=0) 
    filled_normed_conc_attributes = fill_nan(normed_conc_attributes)
    indexlist = np.cumsum(list(map(lambda node_attribute: len(node_attribute), node_attributes)))
    normed_node_attributes = np.array_split(filled_normed_conc_attributes, indexlist[:-1])
    return np.array(normed_node_attributes, dtype=object)

def StandardAttribute(node_attributes):
    conc_attributes = np.asarray(np.concatenate(tuple(i for i in node_attributes),axis=0), dtype=float)                           
    conc_attributes = conc_attributes[:, ~np.all(np.isnan(conc_attributes), axis=0)]
    conc_attributes = conc_attributes[:, ~np.all(conc_attributes == 0, axis=0)] 
    standard_conc_attributes = (conc_attributes - np.nanmean(conc_attributes,axis=0))/np.nanstd(conc_attributes,axis=0)          
    filled_standard_conc_attributes = fill_nan(standard_conc_attributes)
    indexlist = np.cumsum(list(map(lambda node_attribute: len(node_attribute), node_attributes)))
    standard_node_attributes = np.array_split(filled_standard_conc_attributes, indexlist[:-1])
    return np.array(standard_node_attributes, dtype=object)

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
    filled_conc_attributes = fill_nan(conc_attributes)
    indexlist       = np.cumsum(list(map(lambda node_attribute: len(node_attribute), node_attributes)))
    node_attributes = np.array_split(filled_conc_attributes, indexlist[:-1])
    return np.array(node_attributes, dtype=object)   

def cal_node_weights(graph, atoms, cutoff, inner_cutoff, inner_weight, outer_weight):
    """
    Generating cost matrix of wasserstain distance,i.e.,node weights based on given graph representations and certian hyperparameters

    Args:
        graph ([type]): [description]
        atoms ([type]): [description]
        cutoff ([type]): [description]
        inner_cutoff ([type]): [description]
        inner_weight ([type]): [description]
        outer_weight ([type]): [description]
    Returns:
        cost matrix, i.e., atoms weight
    """
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
                if  atoms[edge_i.target].number > 18 and edge_i.target != node_i:  #! this should be target
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

@ray.remote
def multiprocessing_WD(graph_pair_indexs, label_sequences, node_weights):
    return_list = np.array([])
    for graph_pair_index in graph_pair_indexs:
        label1 = graph_pair_index[0]
        label2 = graph_pair_index[1]
        weights_1 = node_weights[label1]
        weights_2 = node_weights[label2+label1]
        costs = ot.dist(label_sequences[label1], label_sequences[label2 + label1], metric='euclidean')
        W_D_i = ot.emd2(weights_1.flatten(), weights_2.flatten(), costs)
        return_list = np.append(return_list, W_D_i)
    # print(W_D_i)
    return return_list

def R_conv_attributes(X, node_features = None, num_iterations=3, sinkhorn=False, enforce_continuous=False): 
    """
    Pairwise computation of the Wasserstein distance between embeddings of the 
    graphs in X.
    args:
        X (List[ig.graphs]): List of graphs
        node_features (array): Array containing the node features for continuously attributed graphs
        num_iterations (int): Number of iterations for the propagation scheme
        sinkhorn (bool): Indicates whether sinkhorn approximation should be used
    """
    # First check if the graphs are continuous vs categorical
    categorical = True
    if enforce_continuous:
        print('Enforce continous flag is on, using CONTINUOUS propagation scheme.')
        categorical = False
    elif node_features is not None:
        print('Continuous node features provided, using CONTINUOUS propagation scheme.')
        categorical = False
    else:
        for g in X:
            if not 'label' in g.vs.attribute_names():
                print('No label attributed to graphs, use degree instead and use CONTINUOUS propagation scheme.')
                categorical = False 
                break
        if categorical:
            print('Categorically-labelled graphs, using CATEGORICAL propagation scheme.')
    # Embed the nodes
    if categorical:
        pass
    else:
        es = ContinuousWeisfeilerLehman()
        node_representations = es.fit_transform(X, node_features=node_features, num_iterations=num_iterations)
    return node_representations

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

if __name__ == "__main__":
    pass
