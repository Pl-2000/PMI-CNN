def maybeExtract(data, patch_size):
    import scipy.io
    try:
        TRAIN = scipy.io.loadmat("../data4/" + data + "_Train_patch_" + str(patch_size)+"_100_PCA" + ".mat")
        TEST  = scipy.io.loadmat("../data4/" + data + "_Test_patch_" + str(patch_size) +"_100_PCA" + ".mat")

    except:
        raise Exception('--data options are: Indian_pines, Salinas, KSC, Botswana OR data files not existed')

    return TRAIN, TEST

def maybeExtract1(data, patch_size):
    import scipy.io
    try:
        TRAIN = scipy.io.loadmat("../data5/" + data + "_Train_patch_" + str(patch_size)+"_100_PCA" + ".mat")
        TEST  = scipy.io.loadmat("../data5/" + data + "_Test_patch_" + str(patch_size) +"_100_PCA" + ".mat")

    except:
        raise Exception('--data options are: Indian_pines, Salinas, KSC, Botswana OR data files not existed')

    return TRAIN, TEST

def maybeExtract_GUass(data, patch_size):
    import scipy.io
    try:
        TRAIN = scipy.io.loadmat("../data2/" + data + "_Train_patch_" + str(patch_size)+"_GUASS_PCA" + ".mat")
        TEST  = scipy.io.loadmat("../data2/" + data + "_Test_patch_" + str(patch_size) +"_GUASS_PCA" + ".mat")

    except:
        raise Exception('--data options are: Indian_pines, Salinas, KSC, Botswana OR data files not existed')

    return TRAIN, TEST



