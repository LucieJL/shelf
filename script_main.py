import os
from datetime import datetime
import hydra
from loaders import load_data, load_model, load_trainer

machine = '14-501-01'
load_from = ''

@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg):
    now = datetime.now().strftime("%m-%dT%H-%M")
    datasets = ['casey2017', 'kerguelen2005', 'kerguelen2014']
    # datasets = ['rosssea2014', 'kerguelen2014']
    #datasets = [d for d in os.listdir('annotations') if os.path.isdir(os.path.join('annotations', d))]

    print(datasets)

    # --- MAIN LOOP  w/ k-cross ---
    for i, test_set in enumerate(datasets):
        print(f'TESTING ON {test_set.upper()}')

        log_path = os.path.join(machine, 'logs', cfg.xp_name, now, f'{test_set}')
        if not os.path.isdir(log_path):
            os.makedirs(log_path)

        train_loader, val_loader, test_loader = load_data(cfg, test_set=test_set)
        model = load_model(cfg, log_path, machine, load_from, test_set)
        pl_trainer = load_trainer(cfg, machine, now, test_set)

        pl_trainer.fit(model, train_loader, val_loader)
        pl_trainer.test(model=model, dataloaders=test_loader)
    # ------

if __name__ == "__main__":
    main()