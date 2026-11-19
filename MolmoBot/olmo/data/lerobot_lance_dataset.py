"""MolmoBot adapter over the external lerobot-lancedb reader.

Only model sample conversion and sampler compatibility live here. Lance I/O,
video range reads, decoding, caching and task lookup belong to lerobot-lancedb.
The episode/time/action contract follows the previously validated MXI adapter.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from olmo.data.dataset import Dataset
from olmo.data.iterable_dataset_mixture import IterableDatasetMixture

CAMERAS = ('observation.images.droid_shoulder_light_randomization',
           'observation.images.wrist_camera_zed_mini')


class LeRobotLanceDataset(Dataset):
    def __init__(self, config, split):
        if split != 'train':
            raise NotImplementedError('This MolmoBot Lance adapter supports train only')
        if config.input_window_size != 1 or config.action_horizon != 16:
            raise NotImplementedError('The current contract requires one observation and horizon16')
        if config.weighted_sampling or config.use_point_prompts:
            raise NotImplementedError('Weighted sampling and point prompts are unsupported')
        self.config = config
        self.split = split
        self.root = str(config.data_path)
        validate_data_path(self.root, require_val=False)
        self.info = json.loads((Path(self.root)/'meta/info.json').read_text())
        features = self.info['features']
        state_shape = features.get('observation.state', {}).get('shape', [])
        if (self.info.get('codebase_version') != 'v3.0'
                or features.get('action', {}).get('shape') != [8]
                or len(state_shape) != 1 or state_shape[0] < 8
                or any(key not in features for key in CAMERAS)):
            raise ValueError('Expected LeRobot v3 Franka data: 8D actions, >=8D state and two RGB cameras')
        self.fps = float(self.info['fps'])
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Invalid dataset fps')
        offsets = [(i+1)/self.fps for i in range(16)]
        self.windows = {'action': offsets}
        self.has_action_source = 'action_source' in features
        if self.has_action_source:
            self.windows['action_source'] = offsets
        self._raw = None
        # Use the dependency's metadata object instead of a second metadata reader.
        meta = self._reader().meta
        columns = meta.episodes.data
        def array(key):
            return columns.column(key).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
        ids, starts, ends, lengths = (array(k) for k in
            ['episode_index', 'dataset_from_index', 'dataset_to_index', 'length'])
        n = int(meta.total_episodes)
        if (n == 0 or not np.array_equal(ids, np.arange(n)) or starts[0] != 0
                or ends[-1] != int(meta.total_frames) or not np.array_equal(starts[1:], ends[:-1])
                or not np.array_equal(ends-starts, lengths) or (lengths <= 0).any()):
            raise ValueError('Inconsistent native episode index')
        splits = self.info.get('splits') or {}
        if splits:
            if split not in splits:
                raise ValueError(f'No {split!r} split declared')
            bounds = splits[split].split(':')
            if len(bounds) != 2:
                raise ValueError('Expected split as start:end episode range')
            begin, end = int(bounds[0] or 0), int(bounds[1] or n)
        else:
            begin, end = 0, n
        if not 0 <= begin < end <= n:
            raise ValueError('Invalid or empty training split')
        self.episode_ids = ids[begin:end]
        self.starts = starts[begin:end]
        self.lengths = lengths[begin:end]
        if (self.lengths < 3).any():
            raise ValueError('Training requires episodes with at least three frames')
        self.processor = config.robot_processor_config.build_preprocessor() if config.robot_processor_config is not None else None
        self.action_dim = self.state_dim = 8

    def _reader(self):
        if self._raw is None:
            from lerobot_lancedb import LanceDBDataset
            self._raw = LanceDBDataset(root=self.root, delta_timestamps=self.windows,
                return_uint8=True, tolerance_s=.002, video_decoder_cache_size=16)
        return self._raw

    def __len__(self):
        return len(self.episode_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_raw'] = None
        return state

    def get(self, index, rng):
        return self.get_batch([index], [rng])[0]

    def get_batch(self, indices, rngs):
        indices = np.asarray(indices, dtype=np.int64)
        if ((indices < 0) | (indices >= len(self))).any():
            raise IndexError('Episode index outside training split')
        steps = [int(rng.randint(0, int(self.lengths[i])-2)) for i,rng in zip(indices,rngs,strict=True)]
        positions = [int(self.starts[i])+step for i,step in zip(indices,steps,strict=True)]
        raw = self._reader().__getitems__(positions)
        out = []
        for i,step,position,item in zip(indices,steps,positions,raw,strict=True):
            episode = int(self.episode_ids[i])
            if int(item['episode_index']) != episode or int(item['index']) != position:
                raise ValueError('Reader returned the wrong episode/frame')
            valid = min(step+17, int(self.lengths[i])-1)-(step+1)
            chunk = np.zeros((16,8), np.float32)
            chunk[:valid] = item['action'][:valid].numpy()
            if self.has_action_source and not np.isin(item['action_source'][:valid].numpy(), [0,1]).all():
                raise ValueError('Unexpected action_source code')
            state = item['observation.state'].numpy()[:8].copy()
            images = []
            for key in CAMERAS:
                image = item[key]
                if image.dtype != torch.uint8 or image.ndim != 3:
                    raise ValueError('Expected uint8 CHW camera frames')
                images.append(image.permute(1,2,0).contiguous().numpy())
            if self.processor is not None:
                state = self.processor.normalize_state(state, 'synthmanip')
                chunk = self.processor.normalize_action(chunk, 'synthmanip')
            out.append(dict(style='demo', image=images, question=item['task'], answers='',
                state=state, action=chunk, action_is_pad=np.arange(16)>=valid,
                metadata=dict(repo_id='synthmanip', split=self.split,
                              canonical_id=f'{self.root}::episode/{episode}', step=step)))
        return out


def build_dataset(name, split):
    from olmo.data.synthmanip_config import synthmanip_config_registry
    config = synthmanip_config_registry.get(name)
    if not name.startswith('synthmanip/') or config is None:
        raise ValueError(f'Expected a registered Synthmanip dataset: {name}')
    dataset = LeRobotLanceDataset(config, split)
    import logging
    logging.getLogger(__name__).info('Direct lerobot-lancedb dataset: %s (%s episodes)',
                                     type(dataset._reader()).__module__, len(dataset))
    return dataset


def validate_data_path(data_path, require_val=True):
    if require_val:
        raise ValueError('The current adapter supports train only; pass --no_val')
    root = Path(data_path)
    if not ((root / "frames.lance").is_dir()
            and (root / "videos.lance").is_dir()
            and (root / "meta/info.json").is_file()):
        raise ValueError("Expected a filesystem LeRobot Lance product with frames.lance, videos.lance, and meta/info.json")


def validate_training_args(args):
    if not args.no_val:
        raise ValueError('The current adapter supports train only; pass --no_val')
    if args.no_stats:
        return
    if not Path(args.stats_path).is_file():
        raise ValueError('Provide precomputed --stats_path or explicitly use --no_stats')
    from olmo.data.robot_processing import RobotProcessorConfig
    stats = RobotProcessorConfig.load(args.stats_path).stats_by_repo.get('synthmanip', {})
    if not {'observation.state', 'action'}.issubset(stats):
        raise ValueError('Precomputed state and action normalization statistics are required')


def _get_many(wrapper,indices,epochs):
    assert wrapper.n_pad==0,'Training probe does not support padded datasets'
    rngs=[np.random.RandomState((wrapper.seed*195172+i+len(wrapper.dataset)*e)%(2**32-1)) for i,e in zip(indices,epochs,strict=True)]
    raw=wrapper.dataset.get_batch(indices,rngs)
    results=[]
    for item,rng in zip(raw,rngs,strict=True):
        item=dict(item)
        if wrapper.weighting:item['weight']=wrapper.weighting
        if wrapper.preprocessor:item=wrapper.preprocessor(item,rng,**wrapper.preprocessor_kwargs)
        results.append(item)
    return results

class BatchedLanceMixture(IterableDatasetMixture):
    def __iter__(self):
        assert self.packer is None,'Packed training needs a separate batching contract'
        worker_info=self.worker_info or torch.utils.data.get_worker_info()
        batch_ix=0;rng=np.random.RandomState(self.seed);counts=np.zeros(len(self.datasets),dtype=np.int64)
        if self.start_index:
            assert self.start_index%self.global_batch_size==0
            for _ in range(self.start_index//self.global_batch_size):
                ix=self._get_next_sources(rng,counts);batch_ix+=1;np.add.at(counts,ix,1)
        shuffled=[(None,None) for _ in self.datasets]
        while True:
            sources=self._get_next_sources(rng,counts)
            if worker_info and batch_ix%worker_info.num_workers!=worker_info.id:
                np.add.at(counts,sources,1);batch_ix+=1;continue
            batch_ix+=1;requests=[]
            for i,source in enumerate(sources):
                count=counts[source];counts[source]+=1
                if (i+self.rank)%self.world_size:continue
                dataset=self.datasets[source];epoch=count//len(dataset)
                old_epoch,order=shuffled[source]
                if epoch!=old_epoch:
                    order=np.arange(len(dataset),dtype=np.int32);np.random.RandomState(self.seed+epoch*1771).shuffle(order);shuffled[source]=(epoch,order)
                requests.append((int(source),int(order[count%len(dataset)]),int(epoch)))
            results=[None]*len(requests)
            for source in sorted({x[0] for x in requests}):
                slots=[i for i,x in enumerate(requests) if x[0]==source]
                wrapper=self.datasets[source]
                if hasattr(wrapper.dataset,'get_batch'):
                    items=_get_many(wrapper,[requests[i][1] for i in slots],[requests[i][2] for i in slots])
                else:items=[wrapper.get(requests[i][1],requests[i][2]) for i in slots]
                for slot,item in zip(slots,items,strict=True):results[slot]=item
            yield from results
