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
    mod.init_optimizer(kvstore=kvstore,
                       optimizer='ccsgd',
                       optimizer_params={
                           'learning_rate': 0.0001,
                           'momentum': 0.0,
                           'wd': 0.0
                       })
    return mod


def benchmark(mod, dry_run=10, iterations=10):
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
    for i in range(dry_run):
        mod.forward(batch, is_train=True)
        mod.backward()
        for output in mod.get_outputs(merge_multi_context=False)[0]:
            output.wait_to_read()
        mod.update()

    tic = time.time()

    # real run
    for i in range(iterations):
        mod.forward(batch, is_train=True)
        mod.backward()
        for output in mod.get_outputs(merge_multi_context=False)[0]:
            output.wait_to_read()
        mod.update()

    return format(mod._exec_group.batch_size / ((time.time() - tic) / iterations), '.2f')


syms = {
    'mnist': (mx.sym.load('mnist.json'), [
        ('data', (64, 1, 28, 28))], [('softmax_label', (64,))]),
    'sockeye': (mx.sym.load('sockeye.json'), [('source', (64, 60)), ('target', (
        64, 60))], [('target_label', (64, 60))])
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark MXNet performance.')
    parser.add_argument('--network', type=str, default='mnist',
                        help='Network to run. Should be one of alexnet|vgg|resnet|inceptionv3|c3d')
    parser.add_argument('--gpus', type=str, default='0',
                        help='The gpus to run on. Multiple gpus should be separated by ,')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Optionally override the default batch size')
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
    # ctx = [mx.gpu(int(i)) for i in args.gpus.strip().split(',')]
    ctx = [mx.cpu()]
    mod = get_module(ctx, sym, provide_data, provide_label,
                     kvstore=args.kv_store, batch_size=args.batch_size)
    print(benchmark(mod), " samples/sec")
