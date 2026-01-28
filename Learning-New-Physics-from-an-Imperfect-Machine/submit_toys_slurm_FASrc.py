import os, sys, json, argparse, glob, time, datetime
import numpy as np

sys.path.insert(1, './utils/')
import NUconfig as nu
import UTILSutils as util



shape_nuisances_id_list = ['scale']
OUTPUT_DIRECTORY = '../output/'

# configuration dictionary
config_json = {
    "make_plots" : True,
    "verbose"    : True,
    "N_Ref"   : 200000,
    "N_Bkg"   : 2000,
    "N_Sig"   : 0,#10,
    "Sig_loc": 6.4,
    "Sig_std": 0.16,
    "is_tail_excess": False,
    "output_directory": OUTPUT_DIRECTORY,
    "shape_nuisances_id":        shape_nuisances_id_list,
    "shape_nuisances_data":      [nu.nuisances_data[k] for k in shape_nuisances_id_list],
    "shape_nuisances_reference": [nu.nuisances_reference[k] for k in shape_nuisances_id_list],
    "shape_nuisances_sigma":     [nu.nuisances_sigma[k] for k in shape_nuisances_id_list], 
    "shape_models":              [nu.shape_model[k] for k in shape_nuisances_id_list],
    "norm_nuisances_data":       nu.nuisances_data['norm'],
    "norm_nuisances_reference":  nu.nuisances_reference['norm'],
    "norm_nuisances_sigma":      nu.nuisances_sigma['norm'],
    "epochs_tau": 30,
    "patience_tau": 10,
    "epochs_delta": 30,
    "patience_delta": 10,
    "novelty_finder_architecture": [1,4,1],
    "novelty_finder_weight_clipping": 9, 
    "correction": "SHAPE", # "SHAPE", "NORM", ""
}

# check for errors in the config_json dictionary
if config_json['correction'] not in ["SHAPE", "NORM", ""]:
    print('Error: "correction" must be one of ["SHAPE", "NORM", ""]')
    exit()

if len(config_json["shape_nuisances_sigma"])!=len(config_json["shape_models"]):
    print('Error: length of "shape_nuisances_sigma" and "shape_models" must be the same.')
    exit()
if len(config_json["shape_nuisances_sigma"])!=len(config_json["shape_nuisances_data"]):
    print('Error: length of "shape_nuisances_sigma" and "shape_nuisances_data" must be the same.')
    exit()
if len(config_json["shape_nuisances_sigma"])!=len(config_json["shape_nuisances_reference"]):
    print('Error: length of "shape_nuisances_sigma" and "shape_nuisances_reference" must be the same.')
    exit()
if config_json["correction"]=='SHAPE' and not len(config_json["shape_models"]):
    print('Error: correction is SHAPE but not specified "shape_models" in the configuration dictionary.')
    exit()

# problem specs                                                                                              
ID = 'Nref'+str(config_json["N_Ref"])+'_Nbkg'+str(config_json["N_Bkg"])+'_Nsig'+str(config_json["N_Sig"])
if config_json["N_Sig"] and (config_json['is_NP2']==False):
    ID += '_Sloc'+str(config_json["sig_loc"])
    ID += '_Sstd'+str(config_json["sig_std"])
if config_json["N_Sig"] and (config_json['is_NP2']==True):
    ID += '_tail-excess'

# add details about the experiment set up to the folder name
correction_details = config_json["correction"]
if config_json["correction"]=='SHAPE':
    correction_details += str(len(config_json["shape_models"]))+'_'
    for i in range(len(config_json["shape_nuisances_id"])):
        key = config_json["shape_nuisances_id"][i]
        if config_json["shape_nuisances_data"][i] !=0:
            correction_details += 'nu'+key+str(config_json["shape_nuisances_data"][i])+'_'
if config_json["correction"]=='NORM' or config_json["correction"]=='SHAPE':
    if config_json["correction"]=='NORM':
        correction_details += '_'
    correction_details += 'nuN'+ str(config_json["norm_nuisances_data"])+'_'
ID+='/'+correction_details 
ID+='_epochsTau'+str(config_json["epochs_tau"])+'_epochsDelta'+str(config_json["epochs_delta"])
ID+='_arc'+str(config_json["novelty_finder_architecture"]).replace(', ', '_').replace('[', '').replace(']', '')+'_wclip'+str(config_json["novelty_finder_weight_clipping"])


config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+ID
if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])

#### launch python script ###########################                                                                            
if __name__ == '__main__':
    parser   = argparse.ArgumentParser()
    parser.add_argument('-p','--pyscript', type=str, help="name of python script to execute", required=True)
    parser.add_argument('-l','--local',    type=int, help='if to be run locally',             required=False, default=0)
    parser.add_argument('-t', '--toys',    type=int, help="number of toys to be processed",   required=False, default=100)
    parser.add_argument('-s', '--firstseed', type=int, help="first seed for toys (if specified the toys are submitted with deterministic seed incresing of one unit)", required=False, default=-1)
    args     = parser.parse_args()
    ntoys    = args.toys
    pyscript = args.pyscript
    firstseed= args.firstseed
    config_json['pyscript'] = pyscript
    json_path = util.create_config_file(config_json, config_json["output_directory"])
    
    pyscript_str = pyscript.replace('.py', '')
    pyscript_str = pyscript_str.replace('_', '/')
    
    if args.local:
        for i in range(ntoys):
            if firstseed>=0:
                seed=i
                seed+=firstseed
            else:
                seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
            os.system("python %s/%s -j %s -s %i"%(os.getcwd(), pyscript, json_path, seed))
    else:
        label = "logs"
        os.system("mkdir %s" %label)
        for i in range(ntoys):
            if firstseed>=0:
                seed=i
                seed+=firstseed
            else:
                seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
            script_sbatch = open("%s/submit_%i.sh" %(label, seed) , 'w')
            script_sbatch.write("#!/bin/bash\n")
            script_sbatch.write("#SBATCH -c 1\n")
            script_sbatch.write("#SBATCH --gpus 1\n")
            script_sbatch.write("#SBATCH -t 0-1:03\n")
            script_sbatch.write("#SBATCH -p gpu\n")
            #script_sbatch.write("#SBATCH -p serial_re\n")                                                                       
            script_sbatch.write("#SBATCH --mem=1000\n")
            script_sbatch.write("#SBATCH -o ./logs/%s"%(pyscript_str)+"_%j.out\n")
            script_sbatch.write("#SBATCH -e ./logs/%s"%(pyscript_str)+"_%j.err\n")
            script_sbatch.write("\n")
            script_sbatch.write("module load python/3.10.9-fasrc01\n")
            script_sbatch.write("module load cuda/11.8.0-fasrc01\n")
            script_sbatch.write("\n")
            script_sbatch.write("python %s/%s -j %s -s %i\n"%(os.getcwd(), pyscript, json_path, seed))
            script_sbatch.close()
            os.system("chmod a+x %s/submit_%i.sh" %(label, seed))
            os.system("sbatch %s/submit_%i.sh"%(label, seed) )

