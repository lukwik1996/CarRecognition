import os
import shutil
import random


def split():
    folders = [f.path for f in os.scandir(directory) if f.is_dir()]
    folderNames = []
    folderPaths = []

    for f in folders:
        # if len([name for name in os.listdir(f) if os.path.isfile(os.path.join(f, name))]) >= 200:
        folderNames.append(os.path.basename(f[:-5]))
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


def remove():

    folderNames = []

    for folder in os.listdir(dataset_train):

        f_path = dataset_train + folder
        if len([name for name in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, name))]) < 500:
            folderNames.append(folder)
            shutil.rmtree(f_path)

    for folder in os.listdir(dataset_test):

        f_path = dataset_test + folder
        if folder in folderNames:
            shutil.rmtree(f_path)



if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", help='path to load full dataset')
    ap.add_argument("-d", "--destination", help='path to save split dataset')
    args = vars(ap.parse_args())
	
    directory = args['source']
    destination = args['destination']

    # split()
    # remove()