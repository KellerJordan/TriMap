"""
animate.py

Python script to produce animated gifs of TriMap training
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'tab10'
cmap = plt.get_cmap()
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import torch
from torchvision import datasets, transforms

import argparse
import sys
import pickle

from trimap import Wrapper as TriMap


def save_viz(dataset='mnist2500', optim='sgd', num_iters=1000, lr=None, animated=True):
    
    X = np.loadtxt('data/%s_X.txt' % dataset)
    labels = np.loadtxt('data/%s_labels.txt' % dataset)
    
    print('Computing TriMap embedding using %s optimizer...' % optim)

    trimap = TriMap(X)
    trimap.load_triplets('models/%s.pkl' % dataset) 
    Y_seq = trimap.embed(num_iters=num_iters, embed_init=0.001*trimap.X[:, :2],
                         optimizer=optim,
                         return_seq=True, verbose=True)
        
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    scatter = ax.scatter([], [], 20, [])
    ax.set_title('%s (epoch 0)' % optim)
    patches = [mpatches.Patch(color=cmap.colors[i], label=str(i)) for i in range(10)]

    def init():
        return scatter,

    def update(i):
        if i % 50 == 0:
            print('Animating, iteration %d / %d' % (i, len(Y_seq)))
        ax.clear()
        plt.legend(handles=patches, loc='upper right')
        ax.scatter(Y_seq[i][:, 0], Y_seq[i][:, 1], 1, labels)
        ax.set_title('%s (epoch %d)' % (optim, i))
        return ax, scatter
    
    if animated:
        anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(Y_seq), interval=50)
        path = 'animations/%s.gif' % optim
        print('Saving animation as %s' % path)
        anim.save(path, writer='imagemagick', fps=30)
    
    else:
        Y = Y_seq[-1]
        ax.clear()
        ax.scatter(Y[:, 0], Y[:, 1], 1, labels)
        ax.set_title('%s' % optim)
        path = 'animations/%s.png' % optim
        print('Saving figure as %s' % path)
        plt.savefig(path)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist2500')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--num-iters', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1.0)
    args = parser.parse_args()
    save_viz(args.dataset, args.optimizer, args.num_iters, args.lr)
