# Timed Left/Right Turn and Exit-Bar Design

## Goal

Make left and right intersection maneuvers independent of lane-following after
the entrance bar. The robot will use timed open-loop motion to enter and turn,
then drive straight until the exit bar is close enough to align against. Straight
intersection behavior remains unchanged.

## Left/Right State Flow

Left and right commands use this maneuver sub-state machine:

1. `ENTRY`: publish `TURN_SPEED` with zero angular velocity for
   `TURN_ENTRY_TIME` seconds.
2. `TURN`: publish `TURN_SPEED` and the signed `TURN_ANGULAR` command for
   `TURN_TIME` seconds.
3. `EXIT_STRAIGHT`: publish `TURN_SPEED` with zero angular velocity while
   looking for the exit crosswalk bar.
4. When the entrance has cleared and the exit bar is stable and near, enter
   the existing `EXIT_ALIGN` state.
5. `EXIT_ALIGN` rotates against the bar angle. After the existing stable-frame
   requirement is met, transition to `FOLLOW`.

Lane observations may still be produced for crosswalk masking and debug output,
but they must not select the phase, end the fixed turn, or generate steering
commands during `ENTRY`, `TURN`, or `EXIT_STRAIGHT`.

The existing `MANEUVER_MAX_TIME` fallback remains the final protection against
an indefinitely missed exit bar. Straight commands keep the current
`STRAIGHT` maneuver path and dual-line bridge.

## Configuration

Keep these left/right tuning values:

- `TURN_ENTRY_TIME`: straight travel before starting the turn.
- `TURN_SPEED`: linear speed in all three left/right maneuver phases.
- `TURN_ANGULAR`: absolute angular speed during `TURN`.
- `TURN_TIME`: fixed turn duration, initially `1.6` seconds.

Expose `TURN_TIME` through ROS as `~turn_time`.

Remove these lane-capture settings and supporting runtime state because no lane
observation ends the turn anymore:

- `TURN_CAPTURE_DELAY`
- `TURN_CAPTURE_FRAMES`
- `TURN_TIMEOUT`
- capture hit counters and capture-timeout warning state

## Exit-Bar Recognition

The exit bar can be sampled as a lane in the same frame, so a same-frame lane
model must not automatically veto a geometrically strong crosswalk bar.

A bar is strong enough to override the lane veto only when all of these hold:

- It matches at least `BAR_STRONG_MIN_MATCHED` stripe candidates.
- The matched stripe centers span at least `STRIPE_GROUP_MIN_SPAN` of the image
  width.
- Each match already satisfies `BAR_STRIPE_MIN_ANGLE`, keeping the bar
  transverse to the stripe direction.
- The bar crosses the vehicle axis.

Stripe-to-bar matching accepts a bar near either the top or bottom end of a
stripe. Add the existing conservative top-end tolerances:

- `BAR_STRIPE_TOP_ABOVE_RATIO = 0.14`
- `BAR_STRIPE_TOP_BELOW_RATIO = 0.10`

Increase `STRIPE_LONG_MAX_RATIO` from `0.36` to `0.45` so close exit stripes are
available as evidence. Keep `BAR_ONLY_MAX_ABS_ANGLE = 20.0`; weak or stripe-free
bars still pass through the existing lane-parallel rejection. Do not globally
disable `_bar_matches_lane`.

## Removed Turn Logic

Remove the left/right-only lane-capture helpers and their call sites:

- `turn_lane_capture_valid`
- `turn_side_capture_valid`
- `turn_side_control_target`
- `update_turn_capture_hits`
- lane bridge and PD steering from the left/right `EXIT_APPROACH` path

Rename the post-turn phase from `EXIT_APPROACH` to `EXIT_STRAIGHT` so debug
output describes the actual command.

## Safety and Regression Tests

Tests must cover:

- `ENTRY` changes to `TURN` only after `TURN_ENTRY_TIME`.
- `TURN` changes to `EXIT_STRAIGHT` only after `TURN_TIME`, without lane input.
- `EXIT_STRAIGHT` publishes fixed straight motion even when lane observations
  are present or absent.
- A stable near exit bar changes the state to `EXIT_ALIGN` only after the
  entrance has cleared.
- Straight intersection behavior still uses the dual-line bridge.
- A strong top-end crosswalk bar survives a contaminated same-frame lane model.
- A plain lane boundary without strong stripe evidence is still rejected.
- The overall maneuver timeout and post-exit entry guard still work.

