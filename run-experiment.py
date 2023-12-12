import argparse
import os
import torch

import models
import dataloaders
import experiments

def main():
    parser = argparse.ArgumentParser(
        description='Train modified transformer with sketch data using PyTorch')
    parser.add_argument("experiment_name", help="Reference name of experiment that you want to run")
    parser.add_argument("--id", default="0", help="Experiment signature")
    parser.add_argument("-o", "--output_dir", default="", help="Output directory")
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU ID to run on")
    parser.add_argument("--model_name", help="Model that you want to experiment on")
    parser.add_argument("--data_loader", default='stroke3', help="Data loader name")
    parser.add_argument("--dataset", default=None, help="Input data folder")
    parser.add_argument("-r", "--resume", default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for training (if applicable)")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training (if applicable)")
    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Instantiate experiment
    Experiment = getattr(experiments, args.experiment_name)
    experiment = Experiment(args.id, args.output_dir, device)

    # If the experiment requires a model, load it
    if Experiment.requires_model:
        ModelClass = getattr(models, args.model_name)
        model = ModelClass().to(device)

        DatasetClass = getattr(dataloaders, args.data_loader)
        dataset = DatasetClass(args.dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Resume from checkpoint if specified
        if args.resume and os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

        experiment.setup_model(model, data_loader, args.epochs)

    # Run the experiment
    experiment.compute()

if __name__ == '__main__':
    main()
