import os
from zipfile import ZipFile

basepath='/Users/gracer/Google Drive/HCP_graph/1200/datasets'
if os.path.exists(basepath):
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(basepath)):
          # print(filenames)
          for filename in filenames:
              if filename == 'HCP_S1200_PTNmaps_d15_25_50_100.zip':
                  print('HI')
                  tmppath=os.sep.join([dirpath, filename])
                  with ZipFile(tmppath, 'r') as zipObj:
                     # Get a list of all archived file names from the zip
                     listOfFileNames = zipObj.namelist()
                     print(listOfFileNames)
                     # Iterate over the file names
      #                for fileName in listOfFileNames:
      #                    # Check filename endswith txt
      #                     if fileName.endswith('04.txt'):
      #                         zipObj.extract(fileName, os.path.join(basepath,'temp_txt'))
      #                     if fileName.endswith('09.txt'):
      #                         zipObj.extract(fileName, os.path.join(basepath,'temp_txt'))
      # infile = os.path.join(,'temp_txt')
else:
    print('try again')
