import logging
from time import time
from config import parser
from utils import set_up_logger, set_seed, set_device, train, evaluate
from models import ALL_MODELS, GraphTransformer
from data import CrystalDataset, SatDataset
from torch_geometric.loader import DataLoader
import torch
import os


def main():
    # initial
    args = parser.parse_args()
    args.embedding_dim = args.dim
    args.num_layers = args.layers
    args.batch_norm = False
    saving_path, saving_name = set_up_logger(args)
    
    set_device(args.cuda, args.device)
    set_seed(args.random_seed)
    
    # load data
    logging.info("Loading {} dataset".format(args.dataset))
    start = time()
    dataset = CrystalDataset(args.dataset, args.data_dir, args.data_file, args.train_ratio, args.valid_ratio, args.test_ratio, args.batch_size)
    train_data = SatDataset(dataset.train_data, degree=True, k_hop=2, se="gnn", use_subgraph_edge_attr=True)
    valid_data = SatDataset(dataset.valid_data, degree=True, k_hop=2, se="gnn", use_subgraph_edge_attr=True)
    test_data = SatDataset(dataset.test_data, degree=True, k_hop=2, se="gnn", use_subgraph_edge_attr=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    logging.info("Loading data costs {: .2f}s".format(time() - start))
    
    # build model
    logging.info("Building models")
    start = time()
    # model = ALL_MODELS[args.model](args)
    model = GraphTransformer(
        in_size=args.init_atom_dim,
        num_class=1,
        d_model=args.embedding_dim,
        dim_feedforward=2*args.embedding_dim,
        dropout=0.2,
        abs_pe_dim=20,
        gnn_type="graphsage",
        use_edge_attr=True,
        num_edge_features=1,
        edge_dim=args.embedding_dim,
        k_hop=2,
        se="gnn",
        deg=train_data.deg,
        in_embed=True,
        edge_embed=True,
    )
    logging.info("Building models costs {: .2f}s".format(time() - start))
    
    # define optimizer
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.warm_up:
        lr_steps = (args.lr - 1e-6) / args.warm_up
        decay_factor = args.lr * args.warm_up ** .5
        
        def lr_scheduler(step):
            if step < args.warm_up:
                return 1e-6 + lr_steps * step
            else:
                return decay_factor * step ** -.5
    
    # train process
    best_epoch = 0
    best_metrics = None
    counter = 0

    logging.info("Start training")
    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, lr_scheduler)
        logging.info("Epoch {} | average train loss: {:.4f} | lr: {:.6f}".format(epoch, train_loss, optimizer.param_groups[0]["lr"]))
        valid_loss, valid_metrics = evaluate(valid_loader, model, criterion, split="valid")
        logging.info("Epoch {} | valid loss: {:.4f} | valid metrics: {}".format(epoch, valid_loss, valid_metrics))
        
        if (epoch + 1) % args.eval_freq == 0:
            if not best_metrics or valid_metrics.mae < best_metrics.mae:
                best_epoch = epoch
                best_metrics = valid_metrics
                torch.save(model.cpu().state_dict(), os.path.join(saving_path, "best_model.pth"))
                logging.info("Epoch {} | save best models in {}".format(epoch, saving_path))
                model.cuda()
                counter = 0
            elif args.patience != -1:
                counter += 1
                if counter >= args.patience:
                    logging.info("Early stop at epoch {}".format(epoch))
                    break

    logging.info("Optimization Finished!")

    # load best models
    if not best_metrics:
        torch.save(model.cpu().state_dict(), os.path.join(saving_path, "best_model.pth"))
    else:
        model.load_state_dict(torch.load(os.path.join(saving_path, "best_model.pth")))
        logging.info("Load best models at epoch {} from {}".format(best_epoch, saving_path))

    model.cuda()
    model.eval()

    _, valid_metrics = evaluate(valid_loader, model, criterion, split="valid")
    logging.info("Valid metrics: {}".format(valid_metrics))
    _, test_metrics = evaluate(test_loader, model, criterion, split="test")
    logging.info("Test metrics: {}".format(test_metrics))


if __name__ == "__main__":
    main()
