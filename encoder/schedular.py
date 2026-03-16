import optax

def create_learning_rate_fn(config, base_learning_rate, steps_per_epoch):
    """Creates learning rate schedule with fixed initial LR (until 10 epoch) and linear decay."""
    constant_fn = optax.constant_schedule(value=base_learning_rate)

    decay_epochs = max(config.n_epochs - 10, 1)
    linear_decay_fn = optax.linear_schedule(
        init_value=base_learning_rate,
        end_value=0.,
        transition_steps=decay_epochs * steps_per_epoch
    )

    schedule_fn = optax.join_schedules(
        schedules=[constant_fn, linear_decay_fn],
        boundaries=[10 * steps_per_epoch]
    )

    return schedule_fn