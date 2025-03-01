default_hooks = dict(
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=100),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=True),
)
