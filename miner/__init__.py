from miner import labelbased, softhard, semihard, npair, dist_miner

BATCHMINING_METHODS = {
                        'semihard':semihard,
                        'softhard':softhard,
                        'dist_miner': dist_miner,
                        'labelbased':labelbased,
                        'npair':npair
                       }


def select(batchminername, opt):
    #####
    if batchminername not in BATCHMINING_METHODS: raise NotImplementedError('Batchmining {} not available!'.format(batchminername))

    batchmine_lib = BATCHMINING_METHODS[batchminername]

    return batchmine_lib.BatchMiner(opt)
