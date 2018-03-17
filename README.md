# mxnet-deepmark

[Deepmark](https://github.com/DeepMark/deepmark) on MXNet.

*Feb 16, 2017, Note:* Upstream stopped maintaining this repo.

This forked repo adds CPU specific changes, and new models for testing.
---
## Quickstart

No dataset download is needed, all tests are done using dummy/random generated data. Please refer to benchmark.py for supported models.

```

# inference benchmark for mnist, resnet-20(cifar10), and resnet-50
./bmark.sh --network mnist,resnet-20,resnet-50

# training benchmark for mnist, resnet-20(cifar10), and resnet-50
./bmark.sh --network mnist,resnet-20,resnet-50 --is-train true

# inference benchmark for resnet-50 with different batch sizes
./bmark.sh --network resnet-50 --batch-size 1,8,64,128

```
