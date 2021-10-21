import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

parser = argparse.ArgumentParser(description='Plot ')
parser.add_argument('config_path', type=str)
args = parser.parse_args()

with open(args.config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

plt.figure(figsize=config['figsize'])

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for guide in config['guides']:
    plt.plot([0, 200], [guide, guide], ls=(0, (4, 4)), c='k', lw=1.3)

for model, model_cfg in config['models'].items():
    df = pd.read_csv(os.path.join(config['log_path'], model + '.csv'))
    plt.plot(df['epoch'], 100*df['train_error'], c=model_cfg['color'], ls='dotted')
    plt.plot(df['epoch'], 100*df['test_error'], c=model_cfg['color'], label=model)
    
    if 'text' in model_cfg:
        txt = model_cfg['text']
        plt.text(145, 100* df.iloc[144]['test_error'] + (-1.5 if txt['loc'] == 'bottom' else 0.8), txt['s'])
    
plt.ylim(0, 0.2)
plt.xlim(0, 160)
plt.yticks([0, 5, 10, 20])

plt.ylabel('error (%)')
plt.xlabel('epochs')
plt.legend(loc=3)
plt.savefig(os.path.join(config['log_path'], config['save_name']))