import os
from glob import glob
from tqdm import tqdm
import hydra
import torch
from loaders import load_data
from perch_mlp import PerchMLP

machine = ''
load_from = ''

@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg):
    datasets = ['kerguelen2014']
    print(datasets)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i, set in enumerate(datasets):
        print(f'FOWARDING ON {set.upper()}')
        log_path = os.path.join('infer', cfg.xp_name, set)

        checkpoint_path = os.path.join(log_path, 'checkpoints')
        checkpoint_file = glob(os.path.join(checkpoint_path, '*.ckpt'))  # returns a list
        model = PerchMLP.load_from_checkpoint(checkpoint_file[0],
                                              n_in=cfg.perch_params.output_size,
                                              n_out=len(cfg.train_params.labels),
                                              lr=cfg.train_params.lr,
                                              log_path=log_path
                                              )
        model.to(device)
        model.eval()

        _, _, test_loader = load_data(cfg, test_set=set, infer=True)

        all_y = []
        all_y_hat = []

        with torch.no_grad():
            for X, y in tqdm(test_loader):
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                all_y_hat.append(y_hat.cpu().detach())
                all_y.append(y.cpu().detach())

        all_y = torch.cat(all_y, dim=0)
        all_y_hat = torch.cat(all_y_hat, dim=0)

        torch.save(all_y, os.path.join(log_path, 'infer_targets.pt'))
        torch.save(all_y_hat, os.path.join(log_path, 'infer_outputs.pt'))

if __name__ == '__main__':
    main()
