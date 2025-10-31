## Students need to implement this file.
import argparse
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser(description="Plot histogram of rewards from JSONL file")
parser.add_argument('--input', type=str, required=True, help='Path to input JSONL file with rewards')
parser.add_argument('--output', type=str, required=True, help='Path to output histogram image file')

args = parser.parse_args()

for path in [args.input]:
    print(f"Processing file: {path}")
    rewards = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # rewards.append(data['reward'])
            title = data['continuations'][0]['method']
            y = data['continuations'][0]['normalized_weights']
            plt.hist(y, bins=10, range=(0, 1))
            plt.title(f"Histogram of normalized weights for method: {title}")
            plt.xlabel('Normalized Weight')
            plt.ylabel('Frequency')
            plt.savefig(args.output + f"_{title.replace(' ', '_').replace(';', '_').replace('=', '_')}_prompt_{data['prompt_id']}.png")
            plt.clf()