# G6: Competitive Strategy

*Perspective B: Business Owner*
*Goal: Market positioning insights.*

## Topics

1. **Uniqueness** - What makes this place special
2. **Comparison** - How it compares to competitors
3. **Loyalty** - Customer retention signals

---

## L0 Primitives

### Uniqueness

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `UNIQUENESS_MENTIONED` | bool | true/false | Review mentions uniqueness |
| `UNIQUE_ASPECT` | enum | food/atmosphere/service/concept/location/none | What's unique |
| `DIFFERENTIATION` | enum | generic/somewhat_different/unique/one_of_a_kind | Degree of uniqueness |
| `SIGNATURE_DISH` | bool | true/false | Mentioned must-try item |
| `HARD_TO_FIND` | bool | true/false | Can't get this elsewhere |
| `MEMORABLE` | enum | forgettable/average/memorable/unforgettable | Memorability |

### Comparison

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `COMPETITOR_MENTIONED` | bool | true/false | Mentions another restaurant |
| `COMPARISON_TYPE` | enum | unfavorable/neutral/favorable | How they compare |
| `COMPARED_ASPECT` | enum | price/quality/service/atmosphere/overall | What's compared |
| `COMPETITOR_NAMED` | bool | true/false | Named specific competitor |
| `PREFERENCE_STATED` | enum | prefer_other/no_preference/prefer_this | Preference |
| `SWITCHING_INTENT` | enum | switching_away/considering/staying/switching_to | Switching behavior |

### Loyalty

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `LOYALTY_SIGNAL` | bool | true/false | Shows loyalty indicators |
| `VISIT_FREQUENCY` | enum | first_time/occasional/regular/frequent | How often they visit |
| `RETURN_INTENT` | enum | never/unlikely/likely/definitely | Will they return |
| `RECOMMEND_INTENT` | enum | warn_others/neutral/recommend/strongly_recommend | Would recommend |
| `EMOTIONAL_CONNECTION` | enum | none/low/moderate/high | Emotional attachment |
| `LOYALTY_REASON` | enum | price/quality/service/habit/love/none | Why loyal |

---

## Keywords for Restaurant Selection

### Uniqueness
```
unique, one of a kind, only place, can't find elsewhere, special,
nothing like it, must try, signature dish, famous for, known for,
hidden gem, best kept secret, standout, original, innovative
```

### Comparison
```
better than, compared to, prefer, instead of, reminds me of,
like [restaurant], not as good as, beats, similar to, alternative,
competition, competitor, other places, versus
```

### Loyalty
```
come back, return, regular, favorite, always go, every week,
highly recommend, tell everyone, bring friends, loyal customer,
my go-to, never disappoints, love this place, obsessed, addicted
```
