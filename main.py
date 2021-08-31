import yaml
import argparse
import pickle
import ray
import numpy as np
from skopt.space import Real, Categorical,Integer
from model.Utility import BayOptCv, ClassifySpecies
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
     ml_dict,            #ml setting
     name_hypers,        #bayesian optimization
     dimensions,
     fix_hypers,
     default_para
):
     """
     5-fold cross validation statified by adsorbate
     
     Args : 
     num_cpus     (int)         : number of cpus for machine learning (used for ray job)
     name_hypers  (list)        : names of hyperparameter to be optimized
     dimensions   (skopt object): dimensions (setting) of optimized hyperparameters
     fix_hypers   (list)        : hyperparameters to be fixed
     default_para (dict)        : initial guess or suggested points of hyperparameters
     
     variable : 
     db_graphs         (graphs object): graph representations (connectivity) of initial guess
     db_atoms          (ASE atoms)    : atoms object of initial guess
     node_attributes   (numpy matrix) : primary features or feature vector in the shape of samples*atoms*features
     list_ads_energies (list)         : adsorption energies
     file_names        (list)         : name of structures
     
     """
     db_graphs, db_atoms, node_attributes, list_ads_energies, file_names = load_training_db(ml_dict)
     
     skf=StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
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
               post_type          = ml_dict["post_type"],
               filenames          = train_file_names
               )
          #starting bayesian optimization to minimize likelihood
          res_opt                 = bayoptcv.BayOpt(
          name_hypers             = name_hypers,
          dimensions              = dimensions,
          fix_hypers              = fix_hypers,
          default_para            = default_para
               )
          #prediction with the use of optimized hyperparameters
          test_RMSE, test_pre            = bayoptcv.Predict(
                    test_graphs          = test_db_graphs,
                    test_atoms           = test_db_atoms,
                    test_node_attributes = test_node_attributes,
                    test_target          = test_list_ads_energies,
                    optimized_hypers     = res_opt.x,
                    name_hypers          = name_hypers
               )
          print(test_RMSE)
     print("Cross validation RMSE: ",np.mean(test_RMSE))

def Extrapolation(
     ml_dict,            #ml setting          
     name_hypers,        #bayesian optimization
     dimensions,
     fix_hypers,
     default_para
):
     """
     Training in domain and predict out of domain
     
     Args : 
     num_cpus     (int)         : number of cpus for machine learning (used for ray job)
     name_hypers  (list)        : names of hyperparameter to be optimized
     dimensions   (skopt object): dimensions (setting) of optimized hyperparameters
     fix_hypers   (list)        : hyperparameters to be fixed
     default_para (dict)        : initial guess or suggested points of hyperparameters
     
     variable : 
     db_graphs         (graphs object): graph representations (connectivity) of initial guess
     db_atoms          (ASE atoms)    : atoms object of initial guess
     node_attributes   (numpy matrix) : primary features or feature vector in the shape of samples*atoms*features
     list_ads_energies (list)         : adsorption energies
     file_names        (list)         : name of structures
     
     """
     train_db_graphs, train_db_atoms, train_node_attributes, \
          train_list_ads_energies, train_file_names = load_training_db(ml_dict)
     test_db_graphs, test_db_atoms, test_node_attributes, \
          test_list_ads_energies, test_file_names = load_test_db(ml_dict)
          
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
          post_type          = ml_dict["post_type"],
          filenames          = train_file_names
          )
     
     #starting bayesian optimization to minimize likelihood
     res_opt                 = bayoptcv.BayOpt(
     name_hypers             = name_hypers,
     dimensions              = dimensions,
     fix_hypers              = fix_hypers,
     default_para            = default_para
          )
     print("hyperparameters:" , res_opt.x) 
     #prediction with the use of optimized hyperparameters
     test_RMSE, test_pre            = bayoptcv.Predict(
               test_graphs          = test_db_graphs,
               test_atoms           = test_db_atoms,
               test_node_attributes = test_node_attributes,
               test_target          = test_list_ads_energies,
               optimized_hypers     = res_opt.x,
               name_hypers          = name_hypers
          )
     print("extrapolation RMSE: ", test_RMSE)

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description='Physic-inspired Wassterian Weisfeiler-Lehman Graph Gaussian Process Regression')
     parser.add_argument("--task", type=str, help="type of ML task", choices=["CV5", "Extrapolation"])
     parser.add_argument("--uuid", type=str, help="uuid for ray job in HPC")
     args   = parser.parse_args()
     
     #! load_setting from input.yml
     print("Load ML setting from input.yml")
     with open('input.yml') as f:
          ml_dict = yaml.safe_load(f)
     #print(ml_dict)
        
     #! initialize ray for paralleization
     ray.init(address=os.environ["ip_head"], _redis_password=args.uuid)
     print("Nodes in the Ray cluster:", ray.nodes())

     if args.task == "CV5":
          #! bayesian optimization setting
          cutoff       = Integer(name='cutoff',             low = 1,      high=5)
          inner_cutoff = Integer(name='inner_cutoff',       low = 1,      high=3)
          inner_weight = Real(name='inner_weight',          low = 0.8,    high=1,     prior='uniform')
          outer_weight = Real(name='outer_weight',          low = 0.05,   high=0.15,  prior='uniform')
          noise_level  = Real(name='noise level',           low = 1e-4,   high=1e0,   prior='uniform')
          gpr_gamma    = Real(name='gamma of gpr',          low = 1e0,    high=100,   prior="uniform" )
          gpr_sigma    = Real(name='sigma of gpr',          low = 1e-2,   high=100,   prior="uniform" )

          name_hypers  = ["inner_weight", "outer_weight",'noise_level', "gpr_gamma"]
          dimensions   = [inner_weight,    outer_weight,  noise_level,   gpr_gamma]
          fix_hypers   = {    "cutoff" : 3,
                              "inner_cutoff":1,
                              "gpr_sigma": 1
                              }
          default_para =     [[           0.9,          0.05,          1e-3,              30],
                              [           0.9,          0.1,           1e-1,              20],
                              [           0.9,          0.1,           1e-2,              30],
                              [           0.9,          0.1,           1e-3,              30],
                              [           0.9,          0.15,          1e-3,              30],
                              [           0.9,          0.1,           1e-3,              50],
                              [           0.9,          0.1,           1e-3,              10],
                              [0.8806875258840565, 0.1327365239845043,  0.009517359358533482, 13.927053762282942],
                              [0.941112318815508,  0.1428003912851984,  0.00924305190195764,  13.832911819240726],
                              [0.9310145125207465, 0.13672134599888425, 0.009130469320228592, 13.77103609259054]
          ]

          SCV5(
          ml_dict      = ml_dict,        
          name_hypers  = name_hypers,
          dimensions   = dimensions,
          fix_hypers   = fix_hypers,
          default_para = default_para
          )

     if args.task == "Extrapolation":
          #! extapolation under certain noise level
          noise_level  = 0.027825594022071243
          
          cutoff       = Integer(name='cutoff',             low = 1,      high=5)
          inner_cutoff = Integer(name='inner_cutoff',       low = 1,      high=3)
          inner_weight = Real(name='inner_weight',          low = 0.8,    high=1,     prior='uniform')
          outer_weight = Real(name='outer_weight',          low = 0.05,   high=0.15,  prior='uniform')
          #noise_level  = Real(name='noise level', low = 1e-4,   high=1e0,   prior='uniform')
          gpr_gamma    = Real(name='gamma of gpr',          low = 1e0,    high=100,   prior="uniform" )

          name_hypers    = ["inner_weight", "outer_weight", "gpr_gamma"]
          dimensions     = [inner_weight,    outer_weight,   gpr_gamma ]

          fix_hypers      = { "cutoff" : 3,
                              "inner_cutoff":1,
                              "noise_level": noise_level,
                              "gpr_sigma": 1
                              }

          default_para   =   [[           0.9,          0.05,               30],
                              [           0.9,          0.1,                20],
                              [           0.9,          0.1,                30],
                              [           0.9,          0.1,                30],
                              [           0.9,          0.15,               30],
                              [           0.9,          0.1,                50],
                              [           0.9,          0.1,                10],
                              [0.8806875258840565,      0.1327365239845043, 13.927053762282942],
                              [0.941112318815508,       0.1428003912851984, 13.832911819240726],
                              [0.9310145125207465,      0.13672134599888425,13.77103609259054],
                              [0.9549620018328902,      0.1384413929381768, 16.06832238503717]]
          Extrapolation(
               ml_dict      = ml_dict,
               name_hypers  = name_hypers,
               dimensions   = dimensions,
               fix_hypers   = fix_hypers,
               default_para = default_para
          )
