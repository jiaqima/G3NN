from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data
from models import GCN, GAT, MLP, GenGNN

# Training settings
parser = argparse.ArgumentParser()

# General configs.
parser.add_argument("--dataset", default="cora")
parser.add_argument("--model", default="gcn")
parser.add_argument("--num_labels_per_class", type=int, default=20)
parser.add_argument("--result_path", default="results")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument(
    '--missing_edge',
    action='store_true',
    default=False,
    help='Missing edge in test set.')
parser.add_argument(
    "--epochs", type=int, default=2000, help="Number of epochs to train.")
parser.add_argument(
    "--patience", type=int, default=200, help="Early stopping patience.")
parser.add_argument("--device", default="cuda")
parser.add_argument("--verbose", type=int, default=1, help="Verbose.")

# Common hyper-parameters.
parser.add_argument(
    "--lr", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).")
parser.add_argument(
    "--hidden", type=int, default=64, help="Number of hidden units.")
parser.add_argument(
    "--dropout",
    type=float,
    default=0.5,
    help="Dropout rate (1 - keep probability).")
parser.add_argument("--activation", default="relu")

# GAT hyper-parameters.
parser.add_argument(
    "--num_heads", type=int, default=8, help="Number of heads.")

# Generative model hyper-parameters.
parser.add_argument(
    "--lamda",
    type=float,
    default=1.0,
    help="Lambda coefficient for nll_discriminative.")
parser.add_argument(
    "--neg_ratio", type=float, default=1.0, help="Negative sample ratio.")

# LSM hyper-parameters.
parser.add_argument(
    "--hidden_x",
    type=int,
    default=2,
    help="Number of hidden units for x_enc.")

# SBM hyper-parameters.
parser.add_argument("--p0", type=float, default=0.9, help="p0 in SBM.")
parser.add_argument("--p1", type=float, default=0.1, help="p1 in SBM.")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

data = load_data(
    dataset=args.dataset,
    num_labels_per_class=args.num_labels_per_class,
    missing_edge=args.missing_edge,
    verbose=args.verbose).to(args.device)

model_args = {
    "num_features": data.num_features,
    "num_classes": data.num_classes,
    "hidden_size": args.hidden,
    "dropout": args.dropout,
    "activation": args.activation
}

if args.model == "gcn":
    model = GCN(**model_args)
elif args.model == "gat":
    model_args["num_heads"] = args.num_heads
    model_args["hidden_size"] = int(args.hidden / args.num_heads)
    model = GAT(**model_args)
elif args.model == "mlp":
    model = MLP(**model_args)
else:
    gen_type, post_type = args.model.split("_")

    gen_config = copy.deepcopy(model_args)
    gen_config["type"] = gen_type
    gen_config["neg_ratio"] = args.neg_ratio
    if gen_type == "lsm":
        gen_config["hidden_x"] = args.hidden_x
    if gen_type == "sbm":
        gen_config["p0"] = args.p0
        gen_config["p1"] = args.p1

    post_config = copy.deepcopy(model_args)
    post_config["type"] = post_type
    if post_type == "gat":
        post_config["num_heads"] = args.num_heads
        post_config["hidden_size"] = int(args.hidden / args.num_heads)
    model = GenGNN(gen_config, post_config)

model = model.to(args.device)
optimizer = optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if hasattr(model, "gen"):

    def train_loss_fn(model, data):
        post_y_log_prob = model(data)
        nll_generative = model.gen.nll_generative(data, post_y_log_prob)
        nll_discriminative = F.nll_loss(post_y_log_prob[data.train_mask],
                                        data.y[data.train_mask])
        return nll_generative + args.lamda * nll_discriminative
else:

    def train_loss_fn(model, data):
        return F.nll_loss(
            model(data)[data.train_mask], data.y[data.train_mask])


def val_loss_fn(logits, data):
    return F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()


def train():
    model.train()
    optimizer.zero_grad()
    loss = train_loss_fn(model, data)
    loss.backward()
    optimizer.step()


def test():
    model.eval()
    logits = model(data)
    val_loss = val_loss_fn(logits, data)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return val_loss, accs


# Training.
patience = args.patience
best_val_loss = np.inf
selected_accs = None
for epoch in range(1, args.epochs):
    if patience < 0:
        break
    train()
    val_loss, accs = test()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        selected_accs = accs
        patience = args.patience
        if args.verbose > 0:
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, *accs))
    patience -= 1

# Save results.
if args.verbose < 1:
    result_path = os.path.join(
        args.result_path,
        "%s/nl%d" % (args.dataset, args.num_labels_per_class))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    results = "vacc_%.4f_tacc_%.4f_seed_%d" % (
        selected_accs[1], selected_accs[2], args.seed)
    model_settng = "model_%s_lr_%.6f_h_%03d_l2_%.6f" % (
        args.model, args.lr, args.hidden, args.weight_decay)
    misc_hp = "act_%s_nh_%d_lambda_%.2f_nr_%.2f_hx_%d_p0_%.2f_p1_%.2f" % (
        args.activation, args.num_heads, args.lamda, args.neg_ratio,
        args.hidden_x, args.p0, args.p1)
    fname = os.path.join(result_path,
                         "_".join([results, model_settng, misc_hp]))
    with open(fname, "w") as f:
        pass
