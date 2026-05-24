# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.


MAJOR = 0
MINOR = 5
PATCH = 0
PRE_RELEASE = 'rc0'

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'grove_fsdp'
__contact_names__ = 'Grove-FSDP maintainers'
__contact_emails__ = ''
__homepage__ = ''
__repository_url__ = ''
__download_url__ = ''
__description__ = (
    'Grove-FSDP: a standalone PyTorch FSDP package with zero-copy buffers and RaggedShard planning'
)
__license__ = 'Apache-2.0'
__keywords__ = (
    'deep learning, distributed training, fsdp, ragged shard, pytorch, torch'
)
