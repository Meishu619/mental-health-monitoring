import audmetric
import numpy as np
import os
import random
import torch
import tqdm
import yaml

from scipy.stats import spearmanr

from data import (
    JSTDataset, 
    load_data
)
from loss import (
    CCCLoss
)
from model import (
    create_model
)
from utils import (
    LabelEncoder,
    Standardizer
)

METRICS = {
    "CC": audmetric.pearson_cc,
    "CCC": audmetric.concordance_cc,
    "RHO": lambda x, y: float(spearmanr(x, y).correlation),
    "MAE": audmetric.mean_absolute_error,
    "MSE": audmetric.mean_squared_error,
}

def train_epoch(
    model,
    optimizer,
    loader,
    device,
    loss_fn
):
    model.train()
    model.to(device)
    
    total_loss = 0
    for data in loader:
        data = {
            key: data[key].to(device).float()
            for key in data
        }
        outputs = model(data)
        loss = loss_fn(outputs, data["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data.cpu().numpy()
    total_loss /= len(loader)
    return total_loss


def evaluate(
    model,
    loader,
    device,
    loss_fn,
    labels
):
    model.eval()
    model.to(device)
    
    total_loss = 0
    predictions = []
    targets = []
    for data in loader:
        data = {
            key: data[key].to(device).float()
            for key in data
        }
        with torch.no_grad():
            outputs = model(data)
            loss = loss_fn(outputs, data["labels"])
        total_loss += loss.data.cpu().numpy()
        predictions.append(outputs.cpu().numpy())
        targets.append(data["labels"].cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    results = {
        target: {
            metric: METRICS[metric](targets[:, index], predictions[:, index])
            for metric in METRICS
        }
        for index, target in enumerate(labels)
    }
    total_loss /= len(loader)
    return total_loss, results, predictions


def experiment(cfg):
    experiment_folder = cfg.meta.results_root
    os.makedirs(experiment_folder, exist_ok=True)
    torch.manual_seed(cfg.hparams.seed)
    np.random.seed(cfg.hparams.seed)
    random.seed(cfg.hparams.seed)

    df_train = load_data(os.path.join(cfg.meta.root, "train")).fillna(0)
    df_dev = load_data(os.path.join(cfg.meta.root, "valid")).fillna(0)
    df_test = load_data(os.path.join(cfg.meta.root, "test")).fillna(0)

    train_subjects = set(df_train["ID"].unique())
    dev_subjects = set(df_dev["ID"].unique())
    test_subjects = set(df_test["ID"].unique())

    if train_subjects != dev_subjects:
        print(f"Train w/o dev: {train_subjects - dev_subjects}")
        print(f"Dev w/o train: {dev_subjects - train_subjects}")
    if train_subjects != test_subjects:
        print(f"Train w/o test: {train_subjects - test_subjects}")
        print(f"Test w/o train: {test_subjects - train_subjects}")

    encoder = LabelEncoder(list(train_subjects))
    encoder.to_yaml(os.path.join(experiment_folder, "encoder.subject.yaml"))
    df_train["ID"] = df_train["ID"].apply(encoder.encode)
    df_dev["ID"] = df_dev["ID"].apply(encoder.encode)
    df_test["ID"] = df_test["ID"].apply(encoder.encode)
    train_subjects = list(df_train["ID"].unique())

    zcm_columns = [x for x in df_train.columns if x[:3] == 'zcm']
    zcm_transform = Standardizer(
        mean=df_train[zcm_columns].mean().values,
        std=df_train[zcm_columns].std().values
    )
    zcm_transform.to_yaml(os.path.join(experiment_folder, "transform.zcm.yaml"))

    pcm_columns = [x for x in df_train.columns if x[:3] == 'pim']
    pcm_transform = Standardizer(
        mean=df_train[pcm_columns].mean().values,
        std=df_train[pcm_columns].std().values
    )
    pcm_transform.to_yaml(os.path.join(experiment_folder, "transform.pcm.yaml"))

    speech_columns = [x for x in df_train.columns if x[:6] == 'Neuron']
    speech_transform = Standardizer(
        mean=df_train[speech_columns].mean().values,
        std=df_train[speech_columns].std().values
    )
    speech_transform.to_yaml(os.path.join(experiment_folder, "transform.speech.yaml"))

    train_dataset = JSTDataset(
        df=df_train, 
        target_column=cfg.targets,
        zcm_columns=zcm_columns,
        pcm_columns=pcm_columns,
        speech_columns=speech_columns,
        zcm_transform=zcm_transform,
        pcm_transform=pcm_transform,
        speech_transform=speech_transform
    )
    dev_dataset = JSTDataset(
        df=df_dev, 
        target_column=cfg.targets,
        zcm_columns=zcm_columns,
        pcm_columns=pcm_columns,
        speech_columns=speech_columns,
        zcm_transform=zcm_transform,
        pcm_transform=pcm_transform,
        speech_transform=speech_transform
    )
    test_dataset = JSTDataset(
        df=df_test, 
        target_column=cfg.targets,
        zcm_columns=zcm_columns,
        pcm_columns=pcm_columns,
        speech_columns=speech_columns,
        zcm_transform=zcm_transform,
        pcm_transform=pcm_transform,
        speech_transform=speech_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.hparams.batch_size,
        num_workers=4
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=cfg.hparams.batch_size,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=cfg.hparams.batch_size,
        num_workers=4
    )
    input = train_dataset[0]

    model = create_model(cfg, input, train_subjects)
    # print(model)
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=cfg.hparams.learning_rate,
        momentum=0.9
    )

    best_loss = 1e32
    best_epoch = None
    best_state = None
    best_results = None
    for epoch in tqdm.tqdm(
        range(cfg.hparams.epochs), 
        total=cfg.hparams.epochs, 
        desc="Training",
        disable=True
    ):
        train_loss = train_epoch(
            model=model,
            device=cfg.meta.device,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=CCCLoss()
        )

        dev_loss, dev_results, dev_preds = evaluate(
            model=model,
            device=cfg.meta.device,
            loader=dev_loader,
            loss_fn=CCCLoss(),
            labels=cfg.targets
        )
        print(f"Epoch [{epoch+1}/{cfg.hparams.epochs}]\tTrain:{train_loss:.3f}\tDev:{dev_loss:.3f}")
        torch.save(
            model.cpu().state_dict(),
            os.path.join(experiment_folder, "state.last.pth.tar")
        )
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_results = dev_results
            best_epoch = epoch+1
            best_results["Epoch"] = best_epoch
            torch.save(
                model.cpu().state_dict(),
                os.path.join(experiment_folder, "state.best.pth.tar")
            )
            with open(os.path.join(experiment_folder, "dev.yaml"), "w") as fp:
                yaml.dump(best_results, fp)
    model.load_state_dict(torch.load(os.path.join(experiment_folder, "state.best.pth.tar")))
    test_loss, test_results, test_preds = evaluate(
        model=model,
        device=cfg.meta.device,
        loader=test_loader,
        loss_fn=CCCLoss(),
        labels=cfg.targets
    )
    test_results["Loss"] = float(test_loss)

    with open(os.path.join(experiment_folder, "test.yaml"), "w") as fp:
        yaml.dump(test_results, fp)
    for index, target in enumerate(cfg.targets):
        df_test[f"{target}.pred"] = test_preds[:, index]
    df_test = df_test.drop(zcm_columns + pcm_columns + speech_columns, axis=1)
    df_test.to_csv(os.path.join(experiment_folder, "test.csv"))
    print("Test results:")
    print(yaml.dump(test_results))