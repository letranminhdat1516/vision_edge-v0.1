# Event Publishing – Severity Mapping, Priority, Captions

- Module: [src/service/emergency_notification_dispatcher.py](src/service/emergency_notification_dispatcher.py) → class `HealthcareEventPublisher`

## Severity Mapping (fallbacks if config missing)
- Fall: `high ≥ 0.60`, `medium ≥ 0.40`, else `low`.
- Seizure: `high ≥ 0.50`, `medium ≥ 0.30`, else `low`.

## Mobile Status Mapping
- `high → danger`, `medium → abnormal_behavior`, `low → normal`.

## Priority Level
- `high=4`, `medium=3`, `low=2`; `acknowledged` reduces priority; `resolved=0`.

## Creation Gating
- Compare new event priority vs highest active; create only if `≥ current_max` (or `> low` if none).

## Intelligent Caption
- Optional BLIP-based Vietnamese caption; else static template including `(Tin cậy: {confidence:.0%})`.

Notes (VI): Ánh xạ ngưỡng → mức nguy hiểm, cơ chế ưu tiên để tránh spam, có caption AI tiếng Việt.
