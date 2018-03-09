# mxnet-deepmark

[Deepmark](https://github.com/DeepMark/deepmark) on MXNet.

*Feb 16, 2017, Note:* Upstream stopped maintaining this repo.

This forked repo adds CPU specific changes, and new models for testing.
---
## Quickstart

No dataset download is needed, all tests are done using dummy/random generated data. Please refer to benchmark.py for supported models.

```
cd image+video/

# resnet-50 inference/training with default BS=64
python benchmark.py --network resnet-50
python benchmark.py --network resnet-50 --is-train true

# resnet-50 inference test with different BS
python benchmark.py --network resnet-50 --batch-size 16,32,64,128
python benchmark.py --network resnet-50 --batch-size 16,32,64,128 --is-train true

```
