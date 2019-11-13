import os
from zipfile import ZipFile
import tarfile

basepath='/Users/gracer/Google Drive/HCP_graph/1200'
def py_files(members):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == ".py":
            yield tarinfo




for (dirpath, dirnames, filenames) in os.walk(basepath):
      for filename in filenames:
          if filename.endswith('Netmats.zip'):
              tmppath=os.sep.join([dirpath, filename])
              with ZipFile(tmppath, 'r') as zipObj:
                 # Get a list of all archived file names from the zip
                 listOfFileNames = zipObj.namelist()
                 for name in listOfFileNames:
                     print(name)
                     if name == "HCP_PTN1200/NodeTimeseries_3T_HCP1200_MSMAll_ICAd15_ts2.tar.gz":
                         zipObj.extract(name, os.path.join(basepath,'datasets'))
                         # 'node_timeseries/3T_HCP1200_MSMAll_d15_ts2/889579.txt'
                         tar = tarfile.open(os.path.join(basepath,'datasets',name))
                         for tarinfo in tar:
                             print(tarinfo.name)
            
                         for item in stuff:
                             print(item)
                         tar.close()


                 # Iterate over the file names
  #                for fileName in listOfFileNames:
  #                    # Check filename endswith txt
  #                     if fileName.endswith('04.txt'):
  #                         zipObj.extract(fileName, os.path.join(arglist['BASEPATH'],'temp_txt'))
  #                     if fileName.endswith('09.txt'):
  #                         zipObj.extract(fileName, os.path.join(arglist['BASEPATH'],'temp_txt'))
  # infile = os.path.join(arglist['BASEPATH'],'temp_txt')
