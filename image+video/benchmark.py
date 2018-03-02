from __future__ import absolute_import

import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"

import argparse
import importlib
import mxnet as mx
import numpy as np
import time

try:
    from utils.memonger import search_plan
except:
    import sys
    sys.path.append("../utils/")
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
    'mnist': (mx.sym.load('mnist.json'), [
        ('data', (64, 1, 28, 28))], [('softmax_label', (64,))]),
    'resnet-50': (mx.sym.load('resnet-50.json'), [
        ('data', (64, 3, 224, 224))], [('softmax_label', (64,))]),
    'vgg-reduced': (mx.sym.load('vgg-reduced.json'), [
        ('data', (64, 3, 300, 300))], [('softmax_label', (64,))]),
    'ssd-vgg16': (mx.sym.load('ssd-vgg16.json'), [
        ('data', (64, 3, 300, 300))], [('label', (64, 21, 6))]),
    'sockeye': (mx.sym.load('sockeye.json'), [('source', (64, 60)), ('target', (
        64, 60))], [('target_label', (64, 60))])
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark MXNet performance.')
    parser.add_argument('--network', type=str, default='mnist',
                        help='Network to run. Should be one of alexnet|vgg|resnet|inceptionv3|c3d')
    parser.add_argument('--gpus', type=str, default=None,
                        help='The gpus to run on. Multiple gpus should be separated by ,')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Optionally override the default batch size')
    parser.add_argument('--iterations', type=int, default=10,
                        help='iterations')
    parser.add_argument('--is-train', default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--kv-store', type=str, default='device',
                        choices=['device', 'local_update_cpu',
                                 'local_allreduce_cpu'],
                        help='How data are aggregated over multi-GPUs')
    args = parser.parse_args()

    if args.network in syms:
        sym, provide_data, provide_label = syms[args.network]
    else:
        net = importlib.import_module(args.network)
        sym, provide_data, provide_label = net.get_symbol()
    ctx = [mx.cpu()]
    if args.gpus is not None:
        ctx = [mx.gpu(int(i)) for i in args.gpus.strip().split(',')]
    batches = [args.batch_size] if args.batch_size != None else [
        1, 2, 4, 8, 16, 32, 64, 128, 256]
    for batch in batches:
        mod = get_module(ctx, sym, provide_data, provide_label,
                         kvstore=args.kv_store, batch_size=batch, is_train=args.is_train)
        score = benchmark(mod, iterations=args.iterations,
                          is_train=args.is_train)
        print("network:" + args.network + ", type:" + ("training" if args.is_train else "inference") +
              ", batch_size:" + str(mod._exec_group.batch_size) + ", score:" + str(score))
