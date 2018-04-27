import os
import shutil
import random

directory = "D:\\si\\VMMRdb\\"
destination = "D:\\si\\mini\\"

folders = [f.path for f in os.scandir(directory) if f.is_dir()]
folderNames = []
folderPaths = []

for f in folders:
    if len([name for name in os.listdir(f) if os.path.isfile(os.path.join(f, name))]) >= 50:
        folderNames.append(os.path.basename(f))
        folderPaths.append(f + "\\")

for folder, fname in zip(folderPaths, folderNames):

    fileNum = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    fileNumTest = 0.2 * fileNum
    files = os.listdir(folder)
    random.shuffle(files)
    i = 0

    for name in files:

        if i < fileNumTest:
            dest = destination + "test\\" + fname

            if not os.path.exists(dest):
                os.mkdir(dest)

            shutil.copy(folder + name, dest)

        else:
            dest = destination + "train\\" + fname

            if not os.path.exists(dest):
                os.mkdir(dest)

            shutil.copy(folder + name, dest)

        i += 1

