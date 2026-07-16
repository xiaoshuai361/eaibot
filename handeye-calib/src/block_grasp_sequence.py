from __future__ import absolute_import, print_function


def _log_without_masking(log, message):
    try:
        log(message)
    except Exception:
        # A logging failure must not hide the original hardware failure.
        pass


def run_block_sequence(dry_run, stop_at_pre_grasp, confirm_pump_off,
                       move_pre, move_contact, pump_on, retreat, log):
    """Run the only permitted tagless-block action order.

    Each callback is synchronous and must raise when its action is not
    positively confirmed.  In particular, a pump service timeout is an
    unknown state rather than success.
    """
    if dry_run:
        _log_without_masking(
            log, 'Dry run: no wrist, pump, or arm motion executed.')
        return 'dry_run'

    move_pre()
    if stop_at_pre_grasp:
        _log_without_masking(
            log, 'Stopped at pre-grasp. No pump command was sent in this run.')
        return 'pre_grasp'

    try:
        confirm_pump_off()
    except Exception:
        _log_without_masking(
            log,
            'Pump state is UNKNOWN after pump-OFF confirmation failed; '
            'contact was aborted and the pump may be ON. Recover manually.')
        raise

    move_contact()

    # This state transition deliberately happens before calling the service:
    # a timeout may mean the command reached the controller without a reply.
    pump_on_attempted = True
    try:
        pump_on()
    except Exception:
        if pump_on_attempted:
            _log_without_masking(
                log,
                'Pump state is UNKNOWN and may be ON after pump-on was '
                'attempted. Stop and recover manually.')
        raise

    try:
        retreat()
    except Exception:
        _log_without_masking(
            log,
            'CRITICAL: retreat failed after pump-on; pump may remain ON. '
            'No automatic recovery motion or pump-off command was sent.')
        raise
    return 'grasped'
