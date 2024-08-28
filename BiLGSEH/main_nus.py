from train import *
import argparse
import torch
from utils import logger, write_tensors_to_excel

def param_list(log, config):
    log.info('>>> Configs List <<<')
    log.info('--- Dadaset:{}'.format(config.DATASET))
    log.info('--- SEED:{}'.format(config.SEED))
    log.info('--- Bit:{}'.format(config.HASH_BIT))
    log.info('--- Batch:{}'.format(config.BATCH_SIZE))
    log.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    log.info('--- Lr_TXT:{}'.format(config.LR_TXT))
    log.info('--- LR_MyNet:{}'.format(config.LR_MyNet))

    log.info('--- lambda1:{}'.format(config.lambda1))
    log.info('--- lambda2:{}'.format(config.lambda2))
    log.info('--- beta:{}'.format(config.beta))
    log.info('--- t:{}'.format(config.t))

def main(p1):
    config = set_parameter(p1)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.cuda.set_device(config.GPU_ID)

    logName = config.DATASET + '_' + str(config.HASH_BIT)
    log = logger(logName)
    param_list(log, config)

    operation = train_net(log, config)
    best_it = best_ti = 0

    import os
    dir_path = os.path.join(config.RESULT_DIR, config.DATASET + str(config.HASH_BIT))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    result_dir_path = os.path.join(dir_path,  'Training_results_map.txt') 
    with open(result_dir_path, 'a') as f:
        f.writelines('===================================================================================\n')
        f.writelines(
            '.....Hyper_parameter list:  max_epoch: %d,  hash_bit: %d, lambda1: %.3f, lambda2: %.3f,  beta: %.3f, t: %.3f \n' % (
                config.NUM_EPOCH, config.HASH_BIT, config.lambda1, config.lambda2, config.beta, config.t))

    if config.TRAIN == True:
        for epoch in range(config.NUM_EPOCH):
            coll_BI, coll_BT, coll_sim, record_index = operation.train_hashcode(epoch)
            operation.train_Hashfunc(coll_BI, coll_BT, coll_sim, record_index, epoch)

            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                MAP_I2T, MAP_T2I = operation.performance_eval()
                log.info('mAP@50 I->T: %.3f, mAP@50 T->I: %.3f' % (MAP_I2T, MAP_T2I))

                with open(result_dir_path, 'a') as f:
                    f.writelines(
                        '===================================================================================\n')
                    f.writelines(
                        '.....Map of the Epoch %d test:   map(i->t): %3.3f, map(t->i): %3.3f, average: %3.3f\n' % (
                            epoch + 1, MAP_I2T, MAP_T2I, (MAP_I2T + MAP_T2I) / 2.))

                if (best_it + best_ti) < (MAP_I2T + MAP_T2I):
                    best_it, best_ti = MAP_I2T, MAP_T2I
                    log.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (best_it, best_ti))
                    operation.save_checkpoints()

                log.info('--------------------------------------------------------------------')
    else:
        ckp = config.DATASET + '_' + str(config.HASH_BIT) + 'bits.pth'
        operation.load_checkpoints(ckp)
        MAP_I2T, MAP_T2I = operation.performance_eval()
        log.info('mAP@50 I->T: %.3f, mAP@50 T->I: %.3f' % (MAP_I2T, MAP_T2I))

        P_I2T, R_I2T, P_T2I, R_T2I = operation.performance_eval_PR()
        excel_dir_path = os.path.join(dir_path, 'excel_pr.xlsx')
        write_tensors_to_excel(P_I2T, P_T2I, R_I2T, R_T2I, excel_dir_path)

def set_parameter():
    parser = argparse.ArgumentParser(description='Ours')
    parser.add_argument('--TRAIN', default=True, help='train or test', type=bool)
    parser.add_argument('--DATASET', default='nus-wide', help='mirflickr, mscoco, nus-wide', type=str)

    parser.add_argument('--lambda1', default=1, type=float, help='10')
    parser.add_argument('--lambda2', default=10, type=float, help='1')
    parser.add_argument('--beta', default=0.01, type=float, help='0.01')
    parser.add_argument('--t', default=0.3, help='temperature', type=float)

    parser.add_argument('--LR_IMG', default=0.0001, type=float, help='0.0001')
    parser.add_argument('--LR_TXT', default=0.0001, type=float, help='0.0001')
    parser.add_argument('--LR_MyNet', default=0.0001, type=float, help='0.0001')
    parser.add_argument('--LR_DIS', default=0.0001, type=float, help='0.0001')

    parser.add_argument('--HASH_BIT', default=16, help='code length', type=int)
    parser.add_argument('--BATCH_SIZE', default=512, type=int)
    parser.add_argument('--NUM_EPOCH', default=60, type=int)
    parser.add_argument('--EVAL_INTERVAL', default=10, type=int)
    parser.add_argument('--GPU_ID', default=0, type=int)
    parser.add_argument('--SEED', default=1, type=int)  # Please choose a suitable random seed.
    parser.add_argument('--NUM_WORKERS', default=8, type=int)
    parser.add_argument('--EPOCH_INTERVAL', default=2, type=int)
    parser.add_argument('--MODEL_DIR', default="./checkpoints", type=str)
    parser.add_argument('--RESULT_DIR', default="./result", type=str)

    config = parser.parse_args()
    return config

if __name__ == '__main__':
    main()

