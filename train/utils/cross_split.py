from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold, KFold

NUM_FOLDS = 4

def print_fold_groups(df, group_col, fold_col='fold'):
    print(f"\nFold grouping by '{group_col}':")
    for fold in range(NUM_FOLDS):
        fold_groups = df[df[fold_col] == fold][group_col].unique()
        print(f"Fold {fold}: {list(fold_groups)}")
        print(f"  Number of samples: {len(df[df[fold_col] == fold])}")


def print_kfolds(images_flare, images_urban, images_fire):
    print("=== Flare dataset folds ===")
    print_fold_groups(images_flare, 'continent')

    print("\n=== Urban dataset folds ===")
    print_fold_groups(images_urban, 'city')

    print("\n=== Fire dataset folds (random split) ===")
    for fold in range(NUM_FOLDS):
        count = len(images_fire[images_fire['fold'] == fold])
        print(f"Fold {fold}: {count} samples")


def create_folds(images_flare, images_urban, images_fire, NUM_FOLDS=5):
    # ---- FLARE (group by continent) ----
    gkf_flare = GroupKFold(n_splits=NUM_FOLDS)
    images_flare = images_flare.copy()
    images_flare['fold'] = -1

    for fold, (_, val_idx) in enumerate(gkf_flare.split(images_flare, groups=images_flare['continent'])):
        images_flare.loc[val_idx, 'fold'] = fold

    # ---- URBAN (group by city) ----
    gkf_urban = GroupKFold(n_splits=NUM_FOLDS)
    images_urban = images_urban.copy()
    images_urban['fold'] = -1

    for fold, (_, val_idx) in enumerate(gkf_urban.split(images_urban, groups=images_urban['city'])):
        images_urban.loc[val_idx, 'fold'] = fold

    # ---- FIRE (random) ----
    kf_fire = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    images_fire = images_fire.copy()
    images_fire['fold'] = -1

    for fold, (_, val_idx) in enumerate(kf_fire.split(images_fire)):
        images_fire.loc[val_idx, 'fold'] = fold

    print_kfolds(images_flare, images_urban, images_fire)

    return images_flare, images_urban, images_fire