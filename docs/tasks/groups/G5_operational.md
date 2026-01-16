# G5: Operational Efficiency

*Perspective B: Business Owner*
*Goal: Diagnose systemic bottlenecks.*

## Topics

1. **Capacity** - Handle volume and busy times
2. **Execution** - Orders right, promises kept
3. **Consistency** - Same quality over time

---

## L0 Primitives

### Capacity

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `CROWDING_MENTIONED` | bool | true/false | Review mentions crowding/staffing |
| `STAFFING_LEVEL` | enum | severely_understaffed/understaffed/adequate/well_staffed | Staffing assessment |
| `CROWD_LEVEL` | enum | empty/moderate/busy/packed | How crowded was it |
| `SERVICE_DEGRADED` | bool | true/false | Did busyness hurt experience |
| `TIME_OF_VISIT` | enum | off_peak/normal/peak/holiday | When they visited |
| `HANDLED_VOLUME` | enum | collapsed/struggled/managed/handled_well | How they handled volume |

### Execution

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `ORDER_ISSUE` | bool | true/false | Was there an order issue |
| `ISSUE_TYPE` | enum | wrong_item/missing_item/wrong_modification/forgotten/other | Type of issue |
| `PROMISE_MADE` | bool | true/false | Was a promise made |
| `PROMISE_KEPT` | enum | broken/partial/kept | Was promise fulfilled |
| `DESCRIPTION_ACCURATE` | enum | false_advertising/exaggerated/accurate/exceeded | Menu accuracy |
| `RESOLUTION` | enum | none/poor/adequate/good | How issue was resolved |

### Consistency

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `REPEAT_VISIT` | bool | true/false | Is this a repeat visit |
| `COMPARED_TO_PAST` | enum | much_worse/worse/same/better/much_better | Compared to before |
| `QUALITY_DRIFT` | enum | declining/stable/improving | Quality trend |
| `MANAGEMENT_CHANGE` | bool | true/false | Mentioned ownership change |
| `RELIABILITY` | enum | unpredictable/hit_or_miss/mostly_reliable/always_reliable | Reliability |
| `VISIT_COUNT` | enum | first/few/regular/many | How many visits |

---

## Keywords for Restaurant Selection

### Capacity
```
understaffed, short staffed, overwhelmed, too busy, swamped, packed,
couldn't handle, busy night, weekend, friday, saturday, rush hour,
holiday, slammed, stretched thin, well staffed, empty
```

### Execution
```
wrong order, forgot my, never came, missing, not what I ordered,
modification, allergy request ignored, promised, said they would,
as described, not as pictured, false advertising, accurate, delivered
```

### Consistency
```
used to be, went downhill, not like before, inconsistent, hit or miss,
depends on the day, always reliable, every time, first time, regular,
come here often, new management, new owner, changed, declined, improved
```
