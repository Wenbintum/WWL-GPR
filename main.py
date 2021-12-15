import yaml
import argparse
import pickle
import ray
import numpy as np
from skopt.space import Real, Categorical,Integer
from model.Utility import ClassifySpecies
from model.WWL_GPR import BayOptCv
import os, sys
from sklearn.model_selection import KFold,LeaveOneOut,StratifiedKFold

def load_training_db(ml_dict):
     with open(ml_dict["database"]["db_graphs"],'rb') as infile: 
          db_graphs = np.array(pickle.load(infile))
     with open(ml_dict["database"]["db_atoms"],'rb') as infile: 
          db_atoms = np.array(pickle.load(infile),  dtype=object)
     with open(ml_dict["database"]["node_attributes"],'rb') as infile: 
          node_attributes = np.array(pickle.load(infile), dtype=object)
     with open(ml_dict["database"]["target_properties"],'rb') as infile: 
          list_ads_energies = np.array(pickle.load(infile))
     with open(ml_dict["database"]["file_names"],'rb') as infile: 
          file_names = np.array(pickle.load(infile))
     return db_graphs, db_atoms, node_attributes, list_ads_energies, file_names

def load_test_db(ml_dict):
     with open(ml_dict["test_database"]["db_graphs"],'rb') as infile: 
          db_graphs = np.array(pickle.load(infile))
     with open(ml_dict["test_database"]["db_atoms"],'rb') as infile: 
          db_atoms = np.array(pickle.load(infile),  dtype=object)
     with open(ml_dict["test_database"]["node_attributes"],'rb') as infile: 
          node_attributes = np.array(pickle.load(infile), dtype=object)
     with open(ml_dict["test_database"]["target_properties"],'rb') as infile: 
          list_ads_energies = np.array(pickle.load(infile))
     with open(ml_dict["test_database"]["file_names"],'rb') as infile: 
          file_names = np.array(pickle.load(infile))
     return db_graphs, db_atoms, node_attributes, list_ads_energies, file_names

def SCV5(
     ml_dict,            
     opt_dimensions,
     default_para,
     fix_hypers
):
     """task1: 5-fold cross validation for in-domain prediction stratified by adsorbate
     Args:
         ml_dict        ([type]): ML setting
         default_para   ([type]): user defined trials
         opt_dimensions ([type]): dimensions object for skopt
         fix_hypers     ([type]): fixed hyperparameters
     """
     
     db_graphs, db_atoms, node_attributes, list_ads_energies, file_names = load_training_db(ml_dict)
     test_RMSEs = []
     f_times = 1
     skf=StratifiedKFold(n_splits=5, random_state=25, shuffle=True)
     for train_index, vali_index in skf.split(list_ads_energies, ClassifySpecies(file_names)):
          train_db_graphs         = db_graphs[train_index]
          train_db_atoms          = db_atoms[train_index]
          train_node_attributes   = node_attributes[train_index]
          train_list_ads_energies = list_ads_energies[train_index]
          train_file_names        = file_names[train_index]

          test_db_graphs          = db_graphs[vali_index]
          test_db_atoms           = db_atoms[vali_index]
          test_node_attributes    = node_attributes[vali_index]
          test_list_ads_energies  = list_ads_energies[vali_index]
          test_file_names         = file_names[vali_index]

          #initialize bayesian optimization
          bayoptcv = BayOptCv(
               classifyspecies    = ClassifySpecies(train_file_names),
               num_cpus           = int(ml_dict["num_cpus"]),
               db_graphs          = train_db_graphs,
               db_atoms           = train_db_atoms,
               node_attributes    = train_node_attributes,
               y                  = train_list_ads_energies,
               drop_list          = None,
               num_iter           = int(ml_dict["num_iter"]),
               pre_data_type      = ml_dict["pre_data_type"],
               filenames          = train_file_names
               )
     
          #starting bayesian optimization to minimize likelihood
          res_opt                 = bayoptcv.BayOpt(
          opt_dimensions          = opt_dimensions,
          default_para            = default_para,
          fix_hypers              = fix_hypers
          )
          print("hyperparameters:" , res_opt.x) 
          
          #prediction with the use of optimized hyperparameters
          test_RMSE, test_pre            = bayoptcv.Predict(
                    test_graphs          = test_db_graphs,
                    test_atoms           = test_db_atoms,
                    test_node_attributes = test_node_attributes,
                    test_target          = test_list_ads_energies,
                    opt_hypers           = dict(zip(opt_dimensions.keys(), res_opt.x))
               )
          print(f"{f_times} fold RMSE: ",test_RMSE)
          test_RMSEs.append(test_RMSE)
          f_times +=1 
     print("Cross validation RMSE: ",np.mean(test_RMSEs))


def SCV5_FHP(
     ml_dict,  
     fix_hypers
):
     """task2: 5-fold cross validation stratified by adsorbate with fixed hyperparameters

     Args:
         ml_dict    ([type]): ML setting
         fix_hypers ([type]): fixed hyperparameters
     """
     
     db_graphs, db_atoms, node_attributes, list_ads_energies, file_names = load_training_db(ml_dict)
     test_RMSEs = []
     f_times = 1
     skf=StratifiedKFold(n_splits=5, random_state=25, shuffle=True)
     for train_index, vali_index in skf.split(list_ads_energies, ClassifySpecies(file_names)):
     
          train_db_graphs         = db_graphs[train_index]
          train_db_atoms          = db_atoms[train_index]
          train_node_attributes   = node_attributes[train_index]
          train_list_ads_energies = list_ads_energies[train_index]
          train_file_names        = file_names[train_index]

          test_db_graphs          = db_graphs[vali_index]
          test_db_atoms           = db_atoms[vali_index]
          test_node_attributes    = node_attributes[vali_index]
          test_list_ads_energies  = list_ads_energies[vali_index]
          test_file_names         = file_names[vali_index]
          
          bayoptcv = BayOptCv(
               classifyspecies    = ClassifySpecies(train_file_names),
               num_cpus           = int(ml_dict["num_cpus"]),
               db_graphs          = train_db_graphs,
               db_atoms           = train_db_atoms,
               node_attributes    = train_node_attributes,
               y                  = train_list_ads_energies,
               drop_list          = None,
               num_iter           = int(ml_dict["num_iter"]),
               pre_data_type      = ml_dict["pre_data_type"],
               filenames          = train_file_names
               )

          test_RMSE, test_pre            = bayoptcv.Predict(
                    test_graphs          = test_db_graphs,
                    test_atoms           = test_db_atoms,
                    test_node_attributes = test_node_attributes,
                    test_target          = test_list_ads_energies,
                    opt_hypers           = fix_hypers
               )
          print(f"{f_times} fold RMSE: ",test_RMSE)
          test_RMSEs.append(test_RMSE)
          f_times +=1 
     print("Cross validation RMSE: ", np.mean(test_RMSEs))


def Extrapolation(
     ml_dict,               
     opt_dimensions,
     default_para,
     fix_hypers
     
):
     """task3: predict alloy when only training on pure metals

     Args:
         ml_dict      ([type]): ML setting
         default_para ([type]): user defined trials
         fix_hypers   ([type]): fixed hyperparameters
     """
     train_db_graphs, train_db_atoms, train_node_attributes, \
          train_list_ads_energies, train_file_names = load_training_db(ml_dict)
     test_db_graphs, test_db_atoms, test_node_attributes, \
          test_list_ads_energies, test_file_names   = load_test_db(ml_dict)
          
     #initialize bayesian optimization
     bayoptcv = BayOptCv(
          classifyspecies    = ClassifySpecies(train_file_names),
          num_cpus           = int(ml_dict["num_cpus"]),
          db_graphs          = train_db_graphs,
          db_atoms           = train_db_atoms,
          node_attributes    = train_node_attributes,
          y                  = train_list_ads_energies,
          drop_list          = None,
          num_iter           = int(ml_dict["num_iter"]),
          pre_data_type      = ml_dict["pre_data_type"],
          filenames          = train_file_names
          )
     
     #starting bayesian optimization to minimize likelihood
     res_opt                 = bayoptcv.BayOpt(
     opt_dimensions          = opt_dimensions,
     default_para            = default_para,
     fix_hypers              = fix_hypers
          )
     print("hyperparameters:" , res_opt.x) 
     
     #prediction with the use of optimized hyperparameters
     test_RMSE, test_pre            = bayoptcv.Predict(
               test_graphs          = test_db_graphs,
               test_atoms           = test_db_atoms,
               test_node_attributes = test_node_attributes,
               test_target          = test_list_ads_energies,
               opt_hypers           = dict(zip(opt_dimensions.keys(), res_opt.x))
          )
     print("extrapolation RMSE: ", test_RMSE)


if __name__ == "__main__":
     
     parser = argparse.ArgumentParser(description='Physic-inspired Wassterien Weisfeiler-Lehman Graph Gaussian Process Regression')
     parser.add_argument("--task", type=str, help="type of ML task", choices=["CV5", "CV5_FHP", "Extrapolation"])
     parser.add_argument("--uuid", type=str, help="uuid for ray job in HPC")
     args   = parser.parse_args()
     
     #! load_setting from input.yml
     print("Load ML setting from input.yml")
     with open('input.yml') as f:
          ml_dict = yaml.safe_load(f)
     #print(ml_dict)
     
     if args.task == "CV5":
          
          #! initialize ray for paralleization
          ray.init(address=os.environ["ip_head"], _redis_password=args.uuid)
          print("Nodes in the Ray cluster:", ray.nodes())
          
          #cutoff       = Integer(name='cutoff',             low = 1,     high=5)
          #inner_cutoff = Integer(name='inner_cutoff',       low = 1,     high=3)
          inner_weight  = Real(name='inner_weight',          low = 0,     high=1,     prior='uniform')
          outer_weight  = Real(name='outer_weight',          low = 0,     high=1,  prior='uniform') 
          gpr_reg       = Real(name='regularization of gpr', low = 1e-3,  high=1e0,   prior='uniform')  
          gpr_len       = Real(name='lengthscale of gpr',    low = 1,     high=100,   prior="uniform")
          #gpr_sigma    = 1 
          edge_s_s      = Real(name='edge weight of surface-surface',      low = 0,      high=1,     prior="uniform")
          edge_s_a      = Real(name='edge weight of surface-adsorbate',    low = 0,      high=1,     prior="uniform")
          edge_a_a      = Real(name='edge weight of adsorbate-adsorbate',  low = 0,      high=1,     prior="uniform")
          
          fix_hypers      = { "cutoff"        : 2,
                              "inner_cutoff"  : 1,
                              "gpr_sigma"     : 1
                              }

          opt_dimensions    =   {    
                              "inner_weight"      : inner_weight,
                              "outer_weight"      : outer_weight,
                              "gpr_reg"           : gpr_reg, 
                              "gpr_len"           : gpr_len,  
                              "edge_s_s"          : edge_s_s,
                              "edge_s_a"          : edge_s_a,
                              "edge_a_a"          : edge_a_a
                              } 

          default_para   =  [[1.0,  0,        0.03,   30,  0,         1,  0],
                              [0.6,  0.0544362754971445, 0.00824480194221483, 11.4733820390901, 0, 1, 0.6994924119498536]
                              ]
          
          SCV5(
               ml_dict        = ml_dict,
               opt_dimensions = opt_dimensions,
               default_para   = default_para,
               fix_hypers     = fix_hypers
          )


     if args.task == "Extrapolation":
          
          #! initialize ray for paralleization
          ray.init(address=os.environ["ip_head"], _redis_password=args.uuid)
          print("Nodes in the Ray cluster:", ray.nodes())
          
          #! extrapolation with certain regularization and lengthscale
          inner_weight  = Real(name='inner_weight',          low = 0,     high=1,     prior='uniform')
          outer_weight  = Real(name='outer_weight',          low = 0,     high=1,  prior='uniform') 
          edge_s_s      = Real(name='edge weight of surface-surface',      low = 0,      high=1,     prior="uniform")
          edge_s_a      = Real(name='edge weight of surface-adsorbate',    low = 0,      high=1,     prior="uniform")
          edge_a_a      = Real(name='edge weight of adsorbate-adsorbate',  low = 0,      high=1,     prior="uniform")
          
          fix_hypers      = { "cutoff"        : 2,
                              "inner_cutoff"  : 1,
                              "gpr_reg"       : 0.0525554561285561, 
                              "gpr_len"       : 13.410525101483,  
                              "gpr_sigma"     : 1
                            }

          opt_dimensions    =   {    
                              "inner_weight"      : inner_weight,
                              "outer_weight"      : outer_weight,
                              "edge_s_s"          : edge_s_s,
                              "edge_s_a"          : edge_s_a,
                              "edge_a_a"          : edge_a_a
                             } 

          default_para   =  [[1.0,  0,                  0, 1,  0],
                             [0.6,  0.0694050764384062, 0, 1, 0.47921973652378186]
                            ]
          
          Extrapolation(
               ml_dict        = ml_dict,
               opt_dimensions = opt_dimensions,
               default_para   = default_para,
               fix_hypers     = fix_hypers
          )
          
          
     if args.task == "CV5_FHP":
          
          #! initialize ray for paralleization
          #ray.init(address=os.environ["ip_head"], _redis_password=args.uuid)
          #print("Nodes in the Ray cluster:", ray.nodes())

          #! running on local desktop or laptop
          ray.init(num_cpus=ml_dict["num_cpus"])
          print("Job running on {} cpus".format(ml_dict["num_cpus"]))
          
          fix_hypers =   {    "cutoff"            : 2,
                              "inner_cutoff"      : 1,
                              "inner_weight"      : 0.6,
                              "outer_weight"      : 0.0544362754971445,
                              "gpr_reg"           : 0.00824480194221483, 
                              "gpr_len"           : 11.4733820390901, 
                              "gpr_sigma"         : 1,              
                              "edge_s_s"          : 0,
                              "edge_s_a"          : 1,
                              "edge_a_a"          : 0.6994924119498536
                         } 

          SCV5_FHP(
               ml_dict, 
               fix_hypers
          )
