# Fast, Private and Protected

## Authors

- **Nicolas Riccieri Gardin Assumpcao**
  - *Institute of Computing, State University of Campinas*
  - Campinas, Brazil
  - Email: n121245@dac.unicamp.br

- **Leandro Villas**
  - *Institute of Computing, State University of Campinas*
  - Campinas, Brazil
  - Email: lvillas@ic.unicamp.br

## Description

The Fast, Private and Protected (FPP) approach is developed to address convergence issues associated with non-iid data in Federated Learning, while also ensuring the privacy of clients' datasets and protecting against model poisoning attacks. FPP achieves these objectives through the following key characteristics:

- Focusing training efforts on clients with the highest loss values, maximizing potential improvement.
- Evaluating the global model using real datasets from clients.
- Implementing recovery of a previous checkpoint of the global model after an attack.
- Utilizing a reputation value to mitigate the participation of malicious clients.
- Employing Secure Aggregation to protect data privacy during gradient aggregation, model performance evaluation, and reputation estimation, without revealing individual gradients.

## Client Execution

To execute a client, run `python Clients/__init__.py` with the following options:

```bash
python Clients/__init__.py -p CLIENT_PORT -s SERVER_URL -i CLIENT_ID --send_noise SEND_NOISE --scale_gradients SCALE_GRADIENTS
```

Arguments:
- `-p CLIENT_PORT`: Port for the client to communicate with the server.
- `-s SERVER_URL`: URL of the server.
- `-i CLIENT_ID`: Client ID.
- `--send_noise SEND_NOISE`: Noise to add to the gradients.
- `--scale_gradients SCALE_GRADIENTS`: Gradient scaling factor.

### Running Clients with Docker

Alternatively, to execute the client using Docker, you can use the image `nriccieri/fpp_client:1.0` and pass the following environment variables:

```bash
docker run -e CLIENT_PORT=<CLIENT_PORT> -e SERVER_URL=<SERVER_URL> -e CLIENT_ID=<CLIENT_ID> -e SEND_NOISE=<SEND_NOISE> -e SCALE_GRADIENTS=<SCALE_GRADIENTS> nriccieri/fpp_client:1.0
```

Replace `<CLIENT_PORT>`, `<SERVER_URL>`, `<CLIENT_ID>`, `<SEND_NOISE>`, and `<SCALE_GRADIENTS>` with your desired values.

## Server Execution

To execute the server, run `python __main__.py` with the following options:

```bash
python __main__.py -n EXPERIMENT_NAME -s STRATEGY --rounds ROUNDS --clients-per-round CLIENTS_PER_ROUND --clients CLIENTS --minimum-clients MINIMUM_CLIENTS --timeout TIMEOUT -p PORT --initial-selection INITIAL_SELECTION --reputation-recover REPUTATION_RECOVER --reputation-penalty REPUTATION_PENALTY --recover-threshold RECOVER_THRESHOLD
```

Arguments:
- `-n EXPERIMENT_NAME`: Name used to identify the experiment.
- `-s STRATEGY`: Training strategy used (options: FedAvg, PoC, TrimmedMean, Median, FPP).
- `--rounds ROUNDS`: Total rounds of Federated Training.
- `--clients-per-round CLIENTS_PER_ROUND`: Number of clients to train in each round.
- `--clients CLIENTS`: Desired number of devices clients.
- `--minimum-clients MINIMUM_CLIENTS`: Minimum number of clients to start training.
- `--timeout TIMEOUT`: Timeout of clients registry in seconds (default=120).
- `-p PORT`: Socket number where the server will listen for clients registry.
- `--initial-selection INITIAL_SELECTION`: Number of clients initially selected for PoC and FPP.
- `--reputation-recover REPUTATION_RECOVER`: Multiplier of the reputation after a successful round (default=1.2).
- `--reputation-penalty REPUTATION_PENALTY`: Multiplier of the reputation after a minor attack (default=0.98).
- `--recover-threshold RECOVER_THRESHOLD`: Threshold tolerance (default=1.15).