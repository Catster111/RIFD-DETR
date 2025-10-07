#!/usr/bin/env python3
"""Reset the RT-DETR keypoint head weights inside a checkpoint.

Usage:
    python tools/reset_keypoint_head.py \
        --config configs/rtdetr/rtdetr_v2_face.yaml \
        --checkpoint-in output/.../last.pth \
        --checkpoint-out output/.../last_keypoint_reset.pth

The script keeps the detector weights, reinitialises the keypoint head for both
`model` and `ema` (if present), drops optimiser/scheduler states, and writes the
updated checkpoint.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig


def load_model(config_path: str):
    """Instantiate the model from config without loading checkpoints."""
    cfg = YAMLConfig(config_path, resume=None)
    model = cfg.model
    return model


def reset_keypoint_head_weights(model):
    if not getattr(model, 'decoder', None):
        raise RuntimeError('Model has no decoder module.')
    head = getattr(model.decoder, 'keypoint_head', None)
    if head is None:
        raise RuntimeError('Model was instantiated without a keypoint head.')
    head._reset_parameters()
    return model


def main():
    parser = argparse.ArgumentParser(description='Reset RT-DETR keypoint head weights in a checkpoint.')
    parser.add_argument('--config', required=True, help='Path to YAML config used to build the model.')
    parser.add_argument('--checkpoint-in', required=True, help='Path to the checkpoint to modify.')
    parser.add_argument('--checkpoint-out', required=True, help='Where to save the updated checkpoint.')
    parser.add_argument('--from-ema', action='store_true', help='Load detector weights from EMA branch if available (default: model branch).')
    args = parser.parse_args()

    print(f'📥 Loading checkpoint: {args.checkpoint_in}')
    state = torch.load(args.checkpoint_in, map_location='cpu')

    model_branch = 'model'
    ema_branch = None
    if args.from_ema and 'ema' in state and isinstance(state['ema'], dict) and 'module' in state['ema']:
        source_weights = state['ema']['module']
        ema_branch = 'ema'
        print('   → Using EMA weights as source.')
    elif 'model' in state:
        source_weights = state['model']
    elif 'ema' in state and isinstance(state['ema'], dict) and 'module' in state['ema']:
        source_weights = state['ema']['module']
        ema_branch = 'ema'
        print('   → Falling back to EMA weights (model branch missing).')
    else:
        raise RuntimeError('Checkpoint does not contain `model` or `ema.module` parameters.')

    model = load_model(args.config)

    def _filter_non_keypoint(state_dict):
        return {k: v for k, v in state_dict.items() if not k.startswith('decoder.keypoint_head')}

    filtered_weights = _filter_non_keypoint(source_weights)
    missing, unexpected = model.load_state_dict(filtered_weights, strict=False)
    if missing or unexpected:
        print(f'⚠️ load_state_dict missing keys: {missing[:5]} (total {len(missing)})')
        print(f'⚠️ load_state_dict unexpected keys: {unexpected[:5]} (total {len(unexpected)})')

    print('🔄 Reinitialising keypoint head weights...')
    reset_keypoint_head_weights(model)

    # Update checkpoint branches
    new_state_dict = model.state_dict()
    state['model'] = new_state_dict
    if 'ema' in state and isinstance(state['ema'], dict) and 'module' in state['ema']:
        print('   → Updating EMA branch to match reinitialised head.')
        ema_model = load_model(args.config)
        ema_filtered = _filter_non_keypoint(state['ema']['module'])
        ema_model.load_state_dict(ema_filtered, strict=False)
        ema_model.decoder.keypoint_head.load_state_dict(model.decoder.keypoint_head.state_dict())
        state['ema']['module'] = ema_model.state_dict()

    # Drop optimiser/scheduler states to avoid incompatibilities
    for key in ['optimizer', 'lr_scheduler', 'lr_warmup_scheduler']:
        if key in state:
            print(f'   → Removing {key} state to avoid stale moments.')
            state.pop(key)

    # Reset bookkeeping so training restarts cleanly
    state['last_epoch'] = -1

    print(f'💾 Saving updated checkpoint to: {args.checkpoint_out}')
    torch.save(state, args.checkpoint_out)
    print('✅ Done.')


if __name__ == '__main__':
    main()
