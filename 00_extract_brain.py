import os
from zipfile import ZipFile
import analysis as an
import pandas
import tarfile

basepath='/Users/gracer/Google Drive/HCP/HCP_graph/1200/'
dim=100

tarp=os.path.join(basepath,'groupICA_3T_HCP1200_MSMAll.tar.gz')
zarp=os.path.join(basepath,'HCP_PTN1200','graph_analysis')
narp= os.path.join(basepath,'HCP_PTN1200','graph_analysis')

for (dirpath, dirnames, filenames) in os.walk(basepath):
      for filename in filenames:
          if filename == 'HCP1200_Parcellation_Timeseries_Netmats.zip':
              tmppath=os.sep.join([dirpath, filename])
              with ZipFile(tmppath, 'r') as zipObj:
                 # Get a list of all archived file names from the zip
                 listOfFileNames = zipObj.namelist()
                 for name in listOfFileNames:
                     print(name)
                     if name == "HCP_PTN1200/groupICA_3T_HCP1200_MSMAll.tar.gz":
                         zipObj.extract(name, os.path.join(basepath))
                         tar = tarfile.open(tarp)
                         tar.extractall(narp)
                         tar.close()


def getit(dim, basepath, liist, type):
    tarp=os.path.join(basepath,'HCP_PTN1200','NodeTimeseries_3T_HCP1200_MSMAll_ICAd%s_ts2.tar.gz'%dim)
    zarp=os.path.join(basepath,'HCP_PTN1200','graph_analysis','%s'%type,'node_timeseries', '3T_HCP1200_MSMAll_d%s_ts2'%dim)
    narp= os.path.join(basepath,'HCP_PTN1200','graph_analysis', '%s'%type)
    if os.path.exists(zarp):
        print('this is already opened, check the dim')
    elif os.path.exists(tarp):
        print('the tar exists, need to unzip')
        tar_heel(tarp, liist, narp)
    else:
        print('the tar file needs to be unzipped')
        for (dirpath, dirnames, filenames) in os.walk(basepath):
              for filename in filenames:
                  if filename == 'HCP1200_Parcellation_Timeseries_Netmats.zip':
                      tmppath=os.sep.join([dirpath, filename])
                      with ZipFile(tmppath, 'r') as zipObj:
                         # Get a list of all archived file names from the zip
                         listOfFileNames = zipObj.namelist()
                         for name in listOfFileNames:
                             print(name)
                             if name == "HCP_PTN1200/NodeTimeseries_3T_HCP1200_MSMAll_ICAd%s_ts2.tar.gz"%dim:
                                 zipObj.extract(name, os.path.join(basepath,'datasets'))
                                 tar_heel(tarp, liist, narp)
