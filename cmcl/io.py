import csv
from glob import glob

import numpy as np
import os.path
import pdb
import xml.etree.ElementTree as ET

#=================================================================================================#
class mods_outputs:
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data = {}
        self.cad_profile_subtypes = []
        
    def _get_ncases(self,algorithm):
        algorithm_dir = "%s\%s" % (self.root_dir,algorithm)
        dir_contents  = glob("%s\*" % algorithm_dir)
        
        for path in dir_contents:
            if os.path.isdir(path):
                subdir_contents = glob("%s\*" % path)
                Ncases = 0
                for subdir_path in subdir_contents:
                    if 'case_' in subdir_path:
                        Ncases += 1
                if Ncases > 0: return Ncases
        print('Failed to detect Ncases')
        return -1
        
    def _get_nruns(self,algorithm):
        of_data_key = self._get_data_key(algorithm,'OF')
        if of_data_key not in self.data:
            self._read_objective_function(algorithm)
        return (self.data[of_data_key])['nRun'].size
        
    def _get_data_key(self,algorithm,subtype):
        return "%s_%s" % (algorithm,subtype)
        
    def _read_objective_function(self,algorithm):
        fname = "%s\%s\%s_OF.csv" % (self.root_dir,algorithm,algorithm)
        
        data = np.genfromtxt(fname,names=True,delimiter=',',deletechars='',dtype=None)
        data.dtype.names = [name.lstrip('subtype_') for name in data.dtype.names]
        self.OF_type = data.dtype.names[1]
        data_key = self._get_data_key(algorithm,'OF')
        self.data[data_key] = data
        
    def _read_subtype(self,algorithm,subtype):
        subtype_fname    = "%s\%s\%s_subtype_%s.csv" % (self.root_dir,algorithm,algorithm,subtype)
        subtype_data_key = self._get_data_key(algorithm,subtype)
        subtype_data     = np.loadtxt(subtype_fname,skiprows=1,delimiter=',')
        self.data[subtype_data_key] = subtype_data

        if 'Pressure' in subtype: self.cad_profile_subtypes.append(subtype)
        
        if subtype in self.cad_profile_subtypes: # Also read CAD list for profiles
            cad_fname    = "%s\Initial\cad_model.csv" % self.root_dir 
            cad_data_key = self._get_data_key(algorithm,'cad')
            cad_data     = np.loadtxt(cad_fname,skiprows=0,delimiter=',')
            self.data[cad_data_key] = cad_data
    
    def get_OF_data(self,algorithm):
        of_data_key = self._get_data_key(algorithm,'OF')
        if of_data_key not in self.data:
            self._read_objective_function(algorithm)
        return self.data[of_data_key]
    
    def get_subtype_data(self,algorithm,subtype,cases=None,nbest=None,nlast=None):
        subtype_data_key = self._get_data_key(algorithm,subtype)
        if subtype_data_key not in self.data:
            self._read_subtype(algorithm,subtype)
        all_data = self.data[subtype_data_key]
        if all_data.ndim != 2: pdb.set_trace() # Not setup to handle data of this shape
        Nruns  = self._get_nruns(algorithm)
        if all_data.shape[0] != Nruns: pdb.set_trace() # Not setup to handle data of this shape
        # Choose one or more runs
        if nbest:
            of_data_key = self._get_data_key(algorithm,'OF')
            if of_data_key not in self.data:
                self._read_objective_function(algorithm)
            
            best_idx = (self.data[of_data_key])[self.OF_type].argsort()[:nbest]
            data_all_cases = all_data[best_idx,:]
        elif nlast:
            data_all_cases = all_data[-nlast:,:]
        else:
            data_all_cases = all_data
        # Choose one or more cases
        if cases is not None:
            if isinstance(cases, list) or isinstance(cases, tuple):
                case_indices = [c - 1 for c in list(cases)]
            else:
                case_indices = [cases - 1]
                
            Ncases = self._get_ncases(algorithm)
            if data_all_cases.ndim == 1:
                if data_all_cases.size != Ncases:
                    data = data_all_cases[:,case_indices]
            elif data_all_cases.ndim == 2:
                
                Ntot = data_all_cases.shape[1]
                if Ntot % Ncases: pdb.set_trace() 
                Nprof = Ntot/Ncases
                for ii,icase in enumerate(case_indices):
                    case_prof = np.squeeze(data_all_cases[:,icase*Nprof:(icase+1)*Nprof])
                    if (ii==0):
                        data = case_prof
                    elif (ii==1):
                        if (data.size != case_prof.size): pdb.set_trace()###
                        data = np.append([data],[case_prof],0)
                    else:
                        data = np.append(data,[case_prof],0)
        else:
            data = data_all_cases
            pdb.set_trace()####
        data = np.squeeze(data)
        
        if subtype in self.cad_profile_subtypes:
            cad_data_key = self._get_data_key(algorithm,'cad')
            cad_data     = self.data[cad_data_key]
            # Fudge to handle one element length difference in CAD values, profiles
            if data.size % cad_data.size != 0:
                if data.size % (cad_data.size-1) != 0:
                    print("Can't reconcile number of elements in CAD data, profile!")
                    pdb.set_trace()
                else:
                    cad_data = cad_data[1:] - (cad_data[1]-cad_data[0])/2
            return {'cad':cad_data, 'profile':data}
        else:
            return data
#=================================================================================================#           

#=================================================================================================#
class mods_input:
    def __init__(self,root_dir):
        raise Exception('Unfinished!')
        self.root_dir = root_dir
        self.fname    = root_dir + "\Working_dir\MoDS_inputs.xml"
        self.read()
    def read(self):
        tree = ET.parse(self.fname)
        root = tree.getroot()
        #namespaces = dict([node for _, node in ET.iterparse(StringIO(my_schema), events=['start-ns'])])
        namespaces = {'': 'http://como.cheng.cam.ac.uk/MoDS'}
        algs = root.findall('algorithms', namespaces)
        #print(top_lev)
        pdb.set_trace()###
    def __str__(self):
        return "MoDS input file read from %s" % self.fname
#=================================================================================================#
