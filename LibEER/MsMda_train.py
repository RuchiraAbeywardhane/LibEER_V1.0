from models.Models import Model
from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import result_log, setup_seed
from Trainer.MsMdaTraining import train
import torch
import torch.optim as optim
import torch.nn as nn

# run this file with
#   seed indep
#   python MsMda_train.py -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_sub_independent_train_val_test_setting -dataset seed_de_lds -dataset_path YOUR\PATH\TO\SEED -batch_size 32 -epochs 200 -lr 0.001 -seed 2024 -onehot
#   seediv indep
#   python MsMda_train.py -metrics acc macro-f1 -metric_choose macro-f1 -setting seediv_sub_independent_train_val_test_setting -dataset seediv_raw -dataset_path YOUR\PATH\TO\SEED_IV -batch_size 32 -epochs 200 -time_window 1 -feature_type de_lds -lr 0.001 -seed 2024 -onehot

def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    best_metrics = []

    # MsMda is a subject-independent (multi-source domain adaptation) model.
    # Each subject's data acts as a source domain; the test subject is the target.
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)

            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            # number_of_source = number of training subjects (each is a source domain)
            number_of_source = len(train_indexes)

            # build per-source datasets
            datasets_train = []
            samples_source = 0
            for i in range(number_of_source):
                src_data, src_label = index_to_data(data_i, label_i, [train_indexes[i]], [], [], args.keep_dim)[:2]
                ds = torch.utils.data.TensorDataset(torch.Tensor(src_data), torch.Tensor(src_label))
                datasets_train.append(ds)
                samples_source = max(samples_source, len(src_data))

            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

            model = Model['MsMda'](
                num_electrodes=channels,
                in_channels=feature_dim,
                num_classes=num_classes,
                number_of_source=number_of_source
            )

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005)
            criterion = nn.NLLLoss()
            output_dir = make_output_dir(args, "MsMda")

            round_metric = train(
                model=model,
                datasets_train=datasets_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                samples_source=samples_source,
                device=device,
                output_dir=output_dir,
                metrics=args.metrics,
                metric_choose=args.metric_choose,
                optimizer=optimizer,
                batch_size=args.batch_size,
                epochs=args.epochs,
                criterion=criterion
            )
            best_metrics.append(round_metric)

    result_log(args, best_metrics)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
