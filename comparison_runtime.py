"""Atomic, integrity-checked recovery for long comparison training jobs."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from comparison_data import digest, file_hash, write_json


def limit_gpu_memory(device, gib):
    import torch
    device = torch.device(device)
    if device.type == 'cuda':
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., gib * 1024**3 / total), device)


def training_state(engine):
    import torch
    return {'config': asdict(engine.cfg), 'step': engine.step_number,
            'generated': engine.generated, 'optimizer': engine.optimizer.state_dict(),
            'rng': engine.rng.getstate(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'scheduler': engine.scheduler.state_dict() if hasattr(engine, 'scheduler') else None}


def restore_training_state(engine, state):
    import torch
    if state['config'] != asdict(engine.cfg) or not 0 <= state['step'] <= engine.cfg.steps:
        raise ValueError('training state configuration/step mismatch')
    if not 0 <= state['generated'] <= engine.cfg.max_sampled_tokens:
        raise ValueError('invalid restored action-token budget')
    if (state['scheduler'] is not None) != hasattr(engine, 'scheduler'):
        raise ValueError('training scheduler mismatch')
    if state['scheduler'] is not None:
        engine.scheduler.load_state_dict(state['scheduler'])
    engine.optimizer.load_state_dict(state['optimizer'])
    engine.rng.setstate(state['rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng']:
        if len(state['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('visible CUDA device count changed on resume')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    engine.step_number, engine.generated = state['step'], state['generated']


def verify_training_checkpoint(path, protocol):
    path = Path(path)
    hashes = json.loads((path / 'hashes.json').read_text())
    if not hashes or any(Path(k).is_absolute() or '..' in Path(k).parts for k in hashes):
        raise ValueError('invalid training checkpoint manifest')
    files = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name != 'hashes.json'}
    if files != set(hashes) or any(file_hash(path/k) != v for k, v in hashes.items()):
        raise ValueError('training checkpoint integrity failure')
    meta = json.loads((path / 'meta.json').read_text())
    if meta['protocol_digest'] != digest(protocol):
        raise ValueError('training checkpoint protocol mismatch')
    return meta


def latest_checkpoint(output, protocol):
    name = (Path(output) / 'latest').read_text().strip()
    if not name.startswith('checkpoint-') or '/' in name or '.' in name:
        raise ValueError('invalid training latest pointer')
    path = Path(output) / name
    verify_training_checkpoint(path, protocol)
    return path


def save_training_checkpoint(engine, backend, output, protocol, metrics):
    import torch
    output = Path(output)
    name = f'checkpoint-{engine.step_number:06d}'
    target, temp = output/name, output/(name + f'.tmp-{time.time_ns()}')
    if target.exists():
        raise FileExistsError('refusing checkpoint overwrite')
    temp.mkdir()
    backend.save(temp/'adapter')
    torch.save(training_state(engine), temp/'state.pt')
    write_json(temp/'metrics.json', metrics)
    write_json(temp/'meta.json', {'step': engine.step_number, 'protocol_digest': digest(protocol)})
    write_json(temp/'hashes.json', {str(p.relative_to(temp)): file_hash(p)
                                  for p in sorted(temp.rglob('*')) if p.is_file()})
    os.replace(temp, target)
    pointer = output/'latest.tmp'
    pointer.write_text(name+'\n')
    os.replace(pointer, output/'latest')
    return target


def adapter_alias(output, name, checkpoint):
    """Relative symlinks retain the established adapter-final evaluation CLI."""
    output = Path(output)
    temporary = output/(name+f'.tmp-{time.time_ns()}')
    temporary.symlink_to(Path(checkpoint.name)/'adapter', target_is_directory=True)
    os.replace(temporary, output/name)


def resume_predictions(output, protocol, cases, resume):
    """Resume only a complete JSONL prefix of the identical immutable protocol."""
    output = Path(output)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'protocol.json', protocol)
        return []
    if json.loads((output/'protocol.json').read_text()) != protocol:
        raise ValueError('evaluation resume protocol differs')
    from comparison_data import read_rows
    rows = read_rows(output/'predictions.jsonl') if (output/'predictions.jsonl').exists() else []
    if len(rows)>len(cases):raise ValueError('extra predictions on resume')
    for row, case in zip(rows, cases):
        if row.get('id')!=case.id or row.get('case_digest')!=digest(asdict(case)) or type(row.get('correct')) is not bool:
            raise ValueError('evaluation resume prefix differs')
    if (output/'summary.json').exists():
        report=json.loads((output/'summary.json').read_text())
        if (len(rows)!=len(cases) or report.get('status')!='complete' or
            report.get('protocol_digest')!=digest(protocol) or
            report.get('predictions_sha256')!=file_hash(output/'predictions.jsonl')):
            raise ValueError('completed evaluation integrity failure')
    return rows
