from __future__ import absolute_import

import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"

import argparse
import importlib
import mxnet as mx
import numpy as np
import time
from os.path import dirname, abspath, join

try:
    from utils.memonger import search_plan
except:
    import sys
    maindir = dirname(dirname(abspath(__file__)))
    sys.path.append(join(maindir, "utils"))
    from memonger import search_plan


def get_module(ctx, sym, provide_data, provide_label, batch_size=None, kvstore=None,
               is_train=True, use_memonger=False):
    if use_memonger:
        sym = search_plan(sym, data=data_shapes)
    mod = mx.mod.Module(symbol=sym,
                        data_names=[name for name, _ in provide_data],
                        label_names=[name for name, _ in provide_label],
                        context=ctx)
    if batch_size is not None:
        provide_data = [(name, (batch_size,) + shape[1:])
                        for name, shape in provide_data]
        provide_label = [(name, (batch_size,) + shape[1:])
                         for name, shape in provide_label]
    if is_train:
        mod.bind(data_shapes=provide_data, label_shapes=provide_label,
                 for_training=True, inputs_need_grad=False)
    else:
        mod.bind(data_shapes=provide_data, label_shapes=provide_label,
                 for_training=False, inputs_need_grad=False)

    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    if is_train:
        mod.init_optimizer(kvstore=kvstore,
                           optimizer='ccsgd',
                           optimizer_params={
                               'learning_rate': 0.0001,
                               'momentum': 0.0,
                               'wd': 0.0
                           })
    return mod


def train(mod, batch, iterations):
    for i in range(iterations):
        mod.forward(batch, is_train=True)
        mod.backward()
        for output in mod.get_outputs(merge_multi_context=False)[0]:
            output.wait_to_read()
        mod.update()


def inference(mod, batch, iterations):
    for i in range(iterations):
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs(merge_multi_context=False)[0]:
            output.wait_to_read()


def benchmark(mod, dry_run=10, iterations=10, is_train=True):
    if len(mod._context) == 1:
        ctx = mod._context[0]
    else:
        ctx = mx.cpu()
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=ctx)
            for _, shape in mod.data_shapes]
    label = [mx.nd.array(np.random.randint(1, 100, size=shape), ctx=ctx)
             for _, shape in mod.label_shapes]
    batch = mx.io.DataBatch(data, label)
    # dry run
    if is_train:
        train(mod, batch, dry_run)
    else:
        inference(mod, batch, dry_run)

    tic = time.time()

    # real run
    if is_train:
        train(mod, batch, iterations)
    else:
        inference(mod, batch, iterations)

    return format(mod._exec_group.batch_size * iterations / (time.time() - tic), '.1f')


syms = {
    'mnist': ('mnist.json', [
        ('data', (128, 1, 28, 28))], [('softmax_label', (128,))]),
    'alexnet': ('alexnet.json', [
        ('data', (128, 3, 224, 224))], [('softmax_label', (128,))]),
    'inception-bn': ('inception-bn.json', [
        ('data', (128, 3, 224, 224))], [('softmax_label', (128,))]),
    'inception-v3': ('inception-v3.json', [
        ('data', (128, 3, 299, 299))], [('softmax_label', (128,))]),
    'resnet-20': ('resnet-20.json', [
        ('data', (128, 3, 28, 28))], [('softmax_label', (128,))]),
    'resnet-50': ('resnet-50.json', [
        ('data', (128, 3, 224, 224))], [('softmax_label', (128,))]),
    'resnet-152': ('resnet-152.json', [
        ('data', (128, 3, 224, 224))], [('softmax_label', (128,))]),
    'vgg-16': ('vgg-16.json', [
        ('data', (128, 3, 224, 224))], [('softmax_label', (128,))]),
    'vgg16-reduced': ('vgg16-reduced.json', [
        ('data', (16, 3, 300, 300))], [('softmax_label', (16,))]),
    'lstm_bucketing-old': ('lstm_bucketing.json', [('data', (256, 32)), ('l0_init_c', (256, 200)), ('l1_init_c', (256, 200)), ('l0_init_h', (256, 200)), ('l1_init_h', (
        256, 200))], [('softmax_label', (256, 32))]),
    'lstm_bucketing': ('lstm_bucketing.json', [('data', (32, 60))], [('softmax_label', (32, 60))]),
    'ssd-vgg16': ('ssd-vgg16.json', [
        ('data', (16, 3, 300, 300))], [('label', (16, 58, 6))]),
    'ssd_vgg16_reduced_300-symbol': ('ssd_vgg16_reduced_300-symbol.json', [
        ('data', (32, 3, 300, 300))], [('label', (32, 58, 6))]),
    'sockeye': ('sockeye.json', [('source', (64, 60)), ('target', (
        64, 60))], [('target_label', (64, 60))])
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark MXNet performance.')
    parser.add_argument('--network', type=str, default='mnist',
                        help='Network to run. Should be one of alexnet|vgg|resnet|inceptionv3|c3d')
    parser.add_argument('--gpus', type=str, default=None,
                        help='The gpus to run on. Multiple gpus should be separated by ,')
    parser.add_argument('--batch-size', type=str, default=None,
                        help='Optionally override the default batch size')
    parser.add_argument('--iterations', type=int, default=10,
                        help='iterations')
    parser.add_argument('--dry-run', type=int, default=10,
                        help='dry run iterations before benchmarking')
    parser.add_argument('--is-train', default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--kv-store', type=str, default='device',
                        choices=['device', 'local_update_cpu',
                                 'local_allreduce_cpu'],
                        help='How data are aggregated over multi-GPUs')
    args = parser.parse_args()

    ctx = [mx.cpu()]
    if args.gpus is not None:
        ctx = [mx.gpu(int(i)) for i in args.gpus.strip().split(',')]
    batches = [int(i) for i in args.batch_size.strip().split(
        ',')] if args.batch_size != None else [None]
    models = [i for i in args.network.strip().split(',')]
    for model in models:
        for batch in batches:
            if model in syms:
                symfile, provide_data, provide_label = syms[model]
                sym = mx.sym.load(join(dirname(abspath(__file__)), symfile))
            else:
                net = importlib.import_module(model)
                sym, provide_data, provide_label = net.get_symbol()
            mod = get_module(ctx, sym, provide_data, provide_label,
                             kvstore=args.kv_store, batch_size=batch, is_train=args.is_train)
            score = benchmark(mod, dry_run=args.dry_run, iterations=args.iterations,
                              is_train=args.is_train)
            print("network:" + model + ", type:" + ("training" if args.is_train else "inference") +
                  ", batch_size:" + str(mod._exec_group.batch_size) + ", score:" + str(score))
