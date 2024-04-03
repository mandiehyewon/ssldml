import copy
from criteria import triplet, margin, angular


def select(loss, args, to_optim, batchminer=None, device=None):
    losses = {'triplet': triplet,
            'margin': margin,
            'angular': angular}

    if loss not in losses: raise NotImplementedError('Loss {} not implemented!'.format(loss))

    loss_lib = losses[loss]

    loss_par_dict  = {'args':args}
    
    if loss_lib.REQUIRES_BATCHMINER:
        loss_par_dict['batchminer'] = batchminer
    if loss_lib.REQUIRES_DEVICE:
        loss_par_dict['device'] = device
        
    criterion = loss_lib.Criterion(**loss_par_dict)

    if loss_lib.REQUIRES_OPTIM:
        if hasattr(criterion,'optim_dict_list') and criterion.optim_dict_list is not None:
            to_optim += criterion.optim_dict_list
        else:
            to_optim += [{'params':criterion.parameters(), 'lr':criterion.lr}]

    return criterion, to_optim