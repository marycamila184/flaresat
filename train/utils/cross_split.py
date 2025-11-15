from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold, KFold

NUM_FOLDS = 4

## UTILS - KFOLDS PRINT
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

def split_by_fold(df, patch_col, mask_col, test_size=0.1):
    train_patches, val_patches = [], []
    train_masks, val_masks = [], []

    for f in df['fold'].unique():
        fold_data = df[df['fold'] == f]
        patches = fold_data[patch_col].tolist()
        masks = fold_data[mask_col].tolist()

        t_p, v_p, t_m, v_m = train_test_split(
            patches, masks,
            test_size=test_size,
            random_state=42,
            shuffle=True
        )

        train_patches.extend(t_p)
        train_masks.extend(t_m)
        val_patches.extend(v_p)
        val_masks.extend(v_m)

    return train_patches, val_patches, train_masks, val_masks

## KFOLD CREATION

# Flare: Group by continent
gkf_flare = GroupKFold(n_splits=NUM_FOLDS)
images_flare['fold'] = -1
for fold, (_, val_idx) in enumerate(gkf_flare.split(images_flare, groups=images_flare['continent'])):
    images_flare.loc[val_idx, 'fold'] = fold

# Urban: Group by City
gkf_urban = GroupKFold(n_splits=NUM_FOLDS)
images_urban['fold'] = -1
for fold, (_, val_idx) in enumerate(gkf_urban.split(images_urban, groups=images_urban['city'])):
    images_urban.loc[val_idx, 'fold'] = fold

# Fire: Random
kf_fire = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
images_fire['fold'] = -1
for fold, (_, val_idx) in enumerate(kf_fire.split(images_fire)):
    images_fire.loc[val_idx, 'fold'] = fold

print_kfolds(images_flare, images_urban, images_fire)
