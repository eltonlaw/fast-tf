import argparse
from models import experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--save_step', type=int, default=10)
    params = parser.parse_args()

    result = experiment(params)

    # Save params and result together
    with open(params.log_dir + "/results.txt", "a") as f:
        f.write(str(vars(params)))
        f.write("\n")
        f.write(str(result))
        f.write("\n")
    print(result)
