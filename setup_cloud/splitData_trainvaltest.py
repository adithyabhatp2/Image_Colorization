import sys
import os
import random

if __name__ == '__main__':

    train = 0.6
    val = 0.2
    test = 0.2

    if len(sys.argv) < 2:
        print('Usage: split.py <pathToRootDir containing n00* imagenet folders.> <opt:train fraction def:0.6> <opt val fraction def:0.2>')
        sys.exit(-1)

    rootDir = sys.argv[1]

    if len(sys.argv) >= 4:
        train = float(sys.argv[2])
        val = float(sys.argv[3])

    test = 1.0 - train - val

    print('Processing root dir {}'.format(rootDir))
    print('Train: {}\tVal: {}\tTest{}'.format(train, val, test))

    sourceDirs = os.listdir(rootDir)

    trainDirName = 'train'
    valDirName = 'val'
    testDirName = 'test'

    os.makedirs(os.path.join(rootDir, trainDirName))
    os.makedirs(os.path.join(rootDir, valDirName))
    os.makedirs(os.path.join(rootDir, testDirName))

    for sourceDirName in sourceDirs:
        if os.path.isdir(os.path.join(rootDir,sourceDirName)):
            print('Processing source dir: {}'.format(sourceDirName))
            files = os.listdir(os.path.join(rootDir, sourceDirName))
            for fileName in files:
                if os.path.isfile(os.path.join(rootDir, sourceDirName, fileName)):
                    randVal = random.uniform(0,1)
                    if randVal < train:
                        os.rename(os.path.join(rootDir, sourceDirName, fileName), os.path.join(rootDir, trainDirName, fileName))
                    elif randVal < train+val:
                        os.rename(os.path.join(rootDir, sourceDirName, fileName), os.path.join(rootDir, valDirName, fileName))
                    else:
                        os.rename(os.path.join(rootDir, sourceDirName, fileName), os.path.join(rootDir, testDirName, fileName))
            os.rmdir(os.path.join(rootDir,sourceDirName))
