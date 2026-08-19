"""
Verify the ablation ladder is a controlled experiment.

Every config in the ladder must be identical in ALL training hyperparameters,
so that a difference in results is attributable to the architecture/loss change
and nothing else. Config drift is how the original A0-vs-A1 comparison became
uninterpretable: A0 carried clip_grad_norm 0.01 and a single-cycle schedule
while A1/A4/A7 used clip 1.0 and a 2-cycle schedule.

  python scripts/check_ladder_alignment.py
"""
import sys, os, yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fields that MUST match across the ladder. Anything here differing is a defect.
CONTROLLED = [
    ('train', 'total_iter'),
    ('train', 'warmup_iter'),
    ('train', 'use_grad_clip'),
    ('train', 'clip_grad_norm'),
    ('train', 'scheduler', 'type'),
    ('train', 'scheduler', 'periods'),
    ('train', 'scheduler', 'restart_weights'),
    ('train', 'scheduler', 'eta_mins'),
    ('train', 'optim_g', 'type'),
    ('train', 'optim_g', 'lr'),
    ('train', 'optim_g', 'betas'),
    ('train', 'mixing_augs', 'mixup'),
    ('train', 'mixing_augs', 'mixup_beta'),
    ('train', 'mixing_augs', 'use_identity'),
    ('train', 'pixel_opt', 'type'),
    ('train', 'pixel_opt', 'loss_weight'),
    ('datasets', 'train', 'batch_size_per_gpu'),
    ('datasets', 'train', 'gt_size'),
    ('datasets', 'train', 'mini_batch_sizes'),
    ('datasets', 'train', 'iters'),
    ('datasets', 'train', 'geometric_augs'),
    ('datasets', 'train', 'dataset_enlarge_ratio'),
    ('datasets', 'train', 'dataroot_gt'),
    ('datasets', 'train', 'dataroot_lq'),
    ('datasets', 'val', 'dataroot_gt'),
    ('datasets', 'val', 'dataroot_lq'),
    ('manual_seed',),
    ('val', 'val_freq'),
    ('network_g', 'n_feat'),
    ('network_g', 'stage'),
    ('network_g', 'num_blocks'),
]

# Fields that are ALLOWED to differ -- these are what the ladder varies.
EXPECTED_TO_DIFFER = [('network_g', 'type'), ('model_type',), ('name',)]

LADDER = [
    ('A0', 'Options/train_Retinexformer_A0_ladder_LOL_v1.yml'),
    ('A1', 'Options/train_FD2RT_A1_LOL_v1_fixed.yml'),
    ('A4', 'Options/train_FD2RT_A4_LOL_v1_fixed.yml'),
    ('A7', 'Options/train_FD2RT_A7_LOL_v1.yml'),
]


def get(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return '<absent>'
        cur = cur[k]
    return cur


def main(ladder=None):
    ladder = ladder or LADDER
    cfgs = {}
    for label, rel in ladder:
        p = os.path.join(_ROOT, rel)
        if not os.path.exists(p):
            print(f'  MISSING: {rel}')
            continue
        with open(p) as f:
            cfgs[label] = yaml.safe_load(f)

    if len(cfgs) < 2:
        print('Need at least two configs to compare.')
        return 1

    labels = list(cfgs)
    print('=' * 78)
    print('LADDER ALIGNMENT CHECK')
    print('=' * 78)
    print('Configs: ' + ', '.join(labels))
    print()

    drift = []
    for path in CONTROLLED:
        vals = {lbl: get(cfgs[lbl], path) for lbl in labels}
        uniq = {repr(v) for v in vals.values()}
        if len(uniq) > 1:
            drift.append((path, vals))

    if not drift:
        print(f'  PASS — all {len(CONTROLLED)} controlled fields match across '
              f'{len(labels)} configs.')
    else:
        print(f'  FAIL — {len(drift)} controlled field(s) differ:\n')
        for path, vals in drift:
            print(f'    {".".join(str(p) for p in path)}')
            for lbl in labels:
                print(f'        {lbl:<4} = {vals[lbl]}')
            print()

    print('-' * 78)
    print('Fields the ladder is SUPPOSED to vary:')
    for path in EXPECTED_TO_DIFFER:
        vals = {lbl: get(cfgs[lbl], path) for lbl in labels}
        joined = '  '.join(f'{lbl}={vals[lbl]}' for lbl in labels)
        print(f'    {".".join(str(p) for p in path):<16} {joined}')

    # auto_resume should be off (or absent) everywhere.
    print('-' * 78)
    risky = [lbl for lbl in labels if cfgs[lbl].get('auto_resume', False)]
    if risky:
        print(f'  WARNING — auto_resume is ON in: {", ".join(risky)}')
    else:
        print('  auto_resume off/absent in all configs (safe default).')

    print()
    print('RESULT: ' + ('LADDER IS CONTROLLED' if not drift
                        else 'LADDER IS CONFOUNDED — fix before training'))
    return 0 if not drift else 1


if __name__ == '__main__':
    sys.exit(main())
