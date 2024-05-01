import argparse
from Model import MLPModel
from Data import Data
from ClientInterface import ClientSet
from Logger import Logger
from Strategies.Strategy_FedAvg import FedAVG
from Strategies.Strategy_Median import Median
from Strategies.Strategy_TrimmedMean import TrimmedMean
from Strategies.Strategy_PoC import PowerOfChoice
from Strategies.Strategy_FPP import FPP


STRATEGIES = [
    'FedAvg', 'PoC', 'TrimmedMean', 'Median', 'FPP'
]


def process_args():
    parser = argparse.ArgumentParser()
    # General Parameters
    parser.add_argument('--experiment-name', '-n', type=str, help='Name used to identify the experiment.')
    parser.add_argument('--strategy', '-s', type=str, help=f'Training strategy used. (list: {", ".join(STRATEGIES)}).')
    parser.add_argument('--rounds', type=int, default=300, help="Total rounds of Federated Training.")
    parser.add_argument('--clients-per-round', type=int, default=6, help="Number of clients to train in each round.")
    parser.add_argument('--clients', type=int, default=12, help="Desired number of devices clients.")
    parser.add_argument('--minimum-clients', type=int, default=12, help='Minimum number of clients to start training.')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout of clients registry in seconds (default=120).')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Socket number where the server will listen for clients registry.')

    parser.add_argument('--initial-selection', type=int, default=9, help='Number of clients initially selected for PoC and FPP.')

    parser.add_argument('--reputation-recover', default=1.2, type=float, help='Multiplier of the reputation after a successful round  (default=1.2).')
    parser.add_argument('--reputation-penalty', default=0.9, type=float, help='Multiplier of the reputation after a minor attack  (default=0.98).')
    parser.add_argument('--recover-threshold', default=1.15, type=float, help='Threshold tolerance  (default=1.15).')

    args = parser.parse_args()

    return args


def main(args):
    # Load Logger
    logger = Logger(args.experiment_name, parameters=vars(args))

    # Load Model with inicial parameters
    model = MLPModel()
    initial_parameters = model.get_initial_model()

    # Load Data
    data = Data()

    # Register Clients
    clients = ClientSet(minimum=args.minimum_clients,
                        desired=args.clients,
                        port=args.port,
                        timeout=args.timeout,
                        logger=logger)
    logger.log_clients(clients)

    if args.strategy == 'FedAvg':
        strategy = FedAVG(
            model_generator=model,
            clients=clients,
            data=data,
            max_rounds=args.rounds,
            logger=logger,
            n_clients=args.clients_per_round,
            initial_parameters=initial_parameters
        )
    elif args.strategy == 'PoC':
        strategy = PowerOfChoice(
            model_generator=model,
            clients=clients,
            data=data,
            max_rounds=args.rounds,
            logger=logger,
            n_clients=args.clients_per_round,
            initial_parameters=initial_parameters,
            initial_selection=args.initial_selection
        )
    elif args.strategy == 'TrimmedMean':
        strategy = TrimmedMean(
            model_generator=model,
            clients=clients,
            data=data,
            max_rounds=args.rounds,
            logger=logger,
            n_clients=args.clients_per_round,
            initial_parameters=initial_parameters
        )
    elif args.strategy == 'Median':
        strategy = Median(
            model_generator=model,
            clients=clients,
            data=data,
            max_rounds=args.rounds,
            logger=logger,
            n_clients=args.clients_per_round,
            initial_parameters=initial_parameters
        )
    elif args.strategy == 'FPP':
        strategy = FPP(
            model_generator=model,
            clients=clients,
            data=data,
            max_rounds=args.rounds,
            logger=logger,
            n_clients=args.clients_per_round,
            initial_selection=args.initial_selection,
            initial_parameters=initial_parameters,
            reputation_recover=args.reputation_recover,
            reputation_penalty=args.reputation_penalty,
            recover_threshold=args.recover_threshold,
        )
    else:
        raise Exception(f"Invalid Strategy: {args.strategy}. Use one of the valid ones: {', '.join(STRATEGIES)}")

    strategy.run()
    logger.save_json()


if __name__ == '__main__':
    args = process_args()
    main(args)
