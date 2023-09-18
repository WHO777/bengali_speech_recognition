seed = 42

model = dict(
    type='SED',
    pretrained='',
    num_classes=264,
    attn_activation='sigmoid',
    backbone=dict(
        type='TimmClassifier',
        name='tf_efficientnet_b3_ns',
        pretrained=True,
        features_only=True))

data = dict(
    seed=seed,
    global_batch_size=16,
    num_folds=5,
    filter_thresh=5,
    upsample_thresh=50,
    train=dict(
        type='BirdCLEF',
        root_dir='/app/datasets/birdclef_2023',
        folds=[0, 2, 3, 4],
        spec_shape=[128, 384],
        normalize=True,
        sample_rate=32000,
        audio_duration=10,
        win_length=2048,
        f_min=20,
        f_max=16000,
        n_mels=None,
        n_fft=2048,
        audio_augments=[
            dict(
                type='TimeShift',
                prob=0.1),
            dict(
                type='GaussianNoise',
                std=[0.0025, 0.025],
                prob=0.35)],
        spec_augments=[
            dict(
                type='TimeFreqMask',
                time_mask=30,
                freq_mask=20,
                prob=0.65)]),
    val=dict(
        type='BirdCLEF',
        root_dir='/app/datasets/birdclef_2023',
        folds=[1],
        spec_shape=[128, 384],
        normalize=True,
        sample_rate=32000,
        audio_duration=10,
        win_length=2048,
        f_min=20,
        f_max=16000,
        n_mels=None,
        n_fft=2048))

optimizer = dict(
    type='Adam',
    lr=1e-3)

scheduler = dict(
    type='CosineAnnealingWarmRestarts',
    T_0=20,
    T_mult=1,
    eta_min=1e-6,
    last_epoch=-1)

metrics = [
    dict(
        type='PaddedCMap',
        padding_factor=5,
        is_sed_output=True),
    dict(
        type='Accuracy',
        is_sed_output=True),
    dict(
        type='AUC_TF',
        num_classes=264,
        is_sed_output=True)]

loss = dict(
    type='SEDLoss')

callbacks = [
    dict(
        type='CSVLoggerCallback'),
    dict(
        type='TensorBoardCallback')]
