#!/usr/bin/env python3
"""Build comprehensive topic selection from full Yelp dataset.

For each of 18 topics, searches the full review dataset for:
- Critical level: Strong evidence (incidents, severe cases)
- High level: Moderate evidence (mentions, concerns)

Output: data/topic_selection/yelp/{topic}.json

Usage:
    .venv/bin/python scripts/build_topic_selection.py
    .venv/bin/python scripts/build_topic_selection.py --topic G1_allergy  # Single topic
    .venv/bin/python scripts/build_topic_selection.py --dry-run  # Show patterns only
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable


@dataclass
class TopicConfig:
    """Configuration for a topic's search patterns."""
    name: str
    group: str
    perspective: str
    description: str
    # Patterns: list of (regex_pattern, weight) tuples
    critical_patterns: list[tuple[str, float]] = field(default_factory=list)
    high_patterns: list[tuple[str, float]] = field(default_factory=list)
    # Optional: custom scorer function
    custom_scorer: Callable | None = None


# =============================================================================
# TOPIC CONFIGURATIONS
# =============================================================================

TOPICS = {
    # -------------------------------------------------------------------------
    # G1: Customer Health
    # -------------------------------------------------------------------------
    "G1_allergy": TopicConfig(
        name="G1_allergy",
        group="G1",
        perspective="Customer Health",
        description="Allergy safety incidents and accommodation",
        critical_patterns=[
            # Must contain "allergy/allergic" context for most patterns
            (r'(epipen|epi-pen|epi pen)', 5.0),
            (r'anaphyla\w*', 5.0),
            (r'allerg\w*.{0,50}(rushed|went|took|drove).{0,10}(the )?(ER\b|emergency room|hospital)', 5.0),
            (r'(rushed|went|took|drove).{0,10}(the )?(ER\b|emergency room|hospital).{0,50}allerg', 5.0),
            (r'allerg\w*.{0,30}(throat|face|tongue).{0,15}(swell|closed|closing)', 4.0),
            (r'allerg\w*.{0,30}couldn.t.{0,10}breathe', 4.0),
            (r'allerg\w*.{0,30}almost died', 5.0),
            (r'(severe|serious|bad)\s+allergic\s+reaction', 4.0),
            (r'(told|asked|informed).{0,30}allerg.{0,30}(still|anyway|but).{0,30}(served|gave|put|had)', 4.0),
            (r'allerg\w*.{0,20}(yelled|screamed|angry|rude|refused)', 3.0),
        ],
        high_patterns=[
            (r'allergic reaction', 2.0),
            (r'allerg\w*.{0,30}(sick|ill|react)', 2.0),
            (r'\b(peanut|nut|shellfish|gluten|dairy)\s+allerg', 1.5),
            (r'cross.?contaminat', 2.0),
            (r'allerg\w*.{0,20}(accommodat|careful|safe)', 1.0),
            (r'\bfood allerg', 1.5),
            (r'allerg\w*.{0,20}(request|ask|told|inform)', 1.0),
            (r'deathly allergic', 3.0),
            (r'severe\w* allerg', 2.0),
        ],
    ),

    "G1_dietary": TopicConfig(
        name="G1_dietary",
        group="G1",
        perspective="Customer Health",
        description="Dietary restriction violations and accommodation",
        critical_patterns=[
            (r'(vegetarian|vegan).{0,30}(served|gave|found).{0,20}(meat|chicken|beef|pork|fish)', 4.0),
            (r'(kosher|halal).{0,30}(violat|not|wasn)', 4.0),
            (r'(gluten.free|celiac).{0,30}(sick|ill|glutened)', 4.0),
            (r'(lied|lying).{0,20}(vegan|vegetarian|gluten|dietary)', 4.0),
        ],
        high_patterns=[
            # Require context - personal declaration or menu reference
            (r'\b(i am|i\'m|we are|my).{0,10}(vegetarian|vegan)', 1.5),
            (r'(vegetarian|vegan).{0,15}(option|menu|friendly|dish)', 1.0),
            (r'(gluten.free|gluten free|GF).{0,15}(option|menu|friendly)', 1.0),
            (r'(kosher|halal).{0,15}(option|menu|certif)', 1.5),
            (r'dietary.{0,15}(restrict|requirement|need)', 1.0),
            (r'(lactose|dairy).{0,10}(free|intoleran)', 1.0),
        ],
    ),

    "G1_hygiene": TopicConfig(
        name="G1_hygiene",
        group="G1",
        perspective="Customer Health",
        description="Food safety and hygiene incidents",
        critical_patterns=[
            (r'food.?poison', 4.0),
            # Require food context for illness
            (r'(sick|ill|vomit|diarrhea).{0,20}(after|from).{0,15}(eat|meal|food|dinner|lunch)', 3.0),
            (r'(food|meal|ate).{0,20}(made|got).{0,10}(me|us|sick|ill)', 3.0),
            (r'(roach|cockroach|rodent|mouse|rat|mice).{0,20}(saw|found|spotted|crawl)', 5.0),
            (r'health.{0,10}(code|inspector|violation|department)', 4.0),
            (r'(raw|undercooked).{0,15}(chicken|pork|meat)', 3.0),
            (r'(mold|moldy|rotten|spoiled).{0,15}(food|bread|produce|dish)', 3.0),
        ],
        high_patterns=[
            (r'(dirty|filthy|unclean|unsanitary)', 2.0),
            (r'(hair|bug|insect).{0,15}(in|found|my).{0,10}(food|dish|plate)', 2.5),
            (r'(bathroom|restroom).{0,20}(dirty|filthy|gross|disgusting)', 1.5),
            (r'(kitchen|cook).{0,15}(dirty|unclean)', 2.0),
            (r'(smell|stink|stank|odor).{0,15}(bad|terrible|awful)', 1.5),
            (r'(sticky|greasy).{0,10}(table|menu|floor)', 1.0),
        ],
    ),

    # -------------------------------------------------------------------------
    # G2: Customer Social
    # -------------------------------------------------------------------------
    "G2_romance": TopicConfig(
        name="G2_romance",
        group="G2",
        perspective="Customer Social",
        description="Romantic dining experiences",
        critical_patterns=[
            (r'(propos\w*|engaged|engagement).{0,30}(here|restaurant|dinner)', 4.0),
            (r'(anniversary|honeymoon).{0,20}(dinner|celebration|special)', 3.0),
            (r'(perfect|amazing|best).{0,15}(date|romantic)', 3.0),
            (r'(ruined|terrible|worst).{0,15}(date|anniversary|romantic)', 3.0),
        ],
        high_patterns=[
            (r'(date night|first date|romantic dinner)', 2.0),
            (r'(boyfriend|girlfriend|husband|wife|partner|spouse).{0,20}(took|brought|surprised)', 1.5),
            (r'(romantic|intimate|cozy).{0,15}(atmosphere|ambiance|setting)', 1.5),
            (r'(valentine|anniversary)', 1.5),
            (r'(candlelit|candle.lit|dim.{0,5}light)', 1.5),
        ],
    ),

    "G2_business": TopicConfig(
        name="G2_business",
        group="G2",
        perspective="Customer Social",
        description="Business and professional dining",
        critical_patterns=[
            (r'(impress|impressed).{0,20}(client|boss|partner|investor)', 3.0),
            (r'(business|client).{0,15}(meeting|dinner|lunch).{0,20}(success|perfect|great)', 3.0),
            (r'(embarrass|disaster).{0,20}(client|business|meeting)', 3.0),
        ],
        high_patterns=[
            (r'(business|client|corporate).{0,15}(dinner|lunch|meeting)', 2.0),
            (r'(colleague|coworker|co-worker|boss|client)', 1.5),
            (r'(professional|upscale|executive)', 1.0),
            (r'(expense|expens\w*).{0,10}(account|report|reimburse)', 1.5),
            (r'(private|quiet).{0,15}(room|booth|table|conversation)', 1.0),
        ],
    ),

    "G2_group": TopicConfig(
        name="G2_group",
        group="G2",
        perspective="Customer Social",
        description="Group and celebration dining",
        critical_patterns=[
            (r'(party|group).{0,10}of.{0,5}(\d{2,}|ten|twelve|fifteen|twenty)', 3.0),
            (r'(birthday|celebration|graduation|shower).{0,20}(perfect|amazing|ruined|disaster)', 3.0),
            (r'(large|big).{0,10}(party|group|gathering).{0,20}(accommodat|handle)', 2.5),
        ],
        high_patterns=[
            (r'(birthday|celebration|party|reunion|graduation)', 1.5),
            (r'(group|party).{0,10}of.{0,5}(\d+|four|five|six|seven|eight)', 1.0),
            (r'(friends|family).{0,15}(gather|dinner|brunch|celebration)', 1.0),
            (r'(reserv\w*).{0,15}(large|big|group|party)', 1.5),
            (r'(private|semi.private).{0,10}(room|event|party)', 1.5),
        ],
    ),

    # -------------------------------------------------------------------------
    # G3: Customer Economic
    # -------------------------------------------------------------------------
    "G3_price_worth": TopicConfig(
        name="G3_price_worth",
        group="G3",
        perspective="Customer Economic",
        description="Value for money assessment",
        critical_patterns=[
            (r'(not|wasn.t|isn.t).{0,10}worth.{0,15}(price|money|cost)', 3.0),
            (r'(overpriced|over.priced|rip.?off)', 3.0),
            (r'(best|great|amazing).{0,10}(value|deal|bang.{0,5}buck)', 2.5),
            (r'(waste|wasted).{0,10}(money|\$|\d+)', 3.0),
        ],
        high_patterns=[
            (r'(price|cost|expensive|cheap|affordable)', 1.0),
            (r'(worth|value).{0,15}(money|price|it)', 1.5),
            (r'(budget|pricey|reasonable|fair).{0,10}(price|cost)', 1.0),
            (r'\$\d+.{0,10}(per|each|person|plate)', 1.0),
            (r'(quality|portion).{0,15}(price|cost|value)', 1.5),
        ],
    ),

    "G3_hidden_costs": TopicConfig(
        name="G3_hidden_costs",
        group="G3",
        perspective="Customer Economic",
        description="Hidden fees and surprise charges",
        critical_patterns=[
            (r'(hidden|surprise|unexpected|sneak).{0,15}(charge|fee|cost|price)', 4.0),
            (r'(didn.t|never).{0,15}(mention|tell|say).{0,15}(charge|fee|cost)', 3.0),
            (r'(auto|automatic).{0,10}(gratuity|tip|service.charge)', 2.5),
            (r'(bill|check).{0,15}(shock|surprise|higher|more)', 2.5),
        ],
        high_patterns=[
            (r'(service|corkage|split|sharing).{0,10}(charge|fee)', 2.0),
            (r'(extra|additional).{0,15}(charge|fee|cost)', 1.5),
            (r'(tip|gratuity).{0,10}(includ|already|added|automatic)', 1.5),
            (r'(fine print|small print)', 2.0),
            (r'(bread|water|rice).{0,15}(charge|extra|\$)', 1.5),
        ],
    ),

    "G3_time_value": TopicConfig(
        name="G3_time_value",
        group="G3",
        perspective="Customer Economic",
        description="Time efficiency and wait times",
        critical_patterns=[
            # Require long waits (30+ minutes or explicit "hour")
            (r'(wait|waited).{0,10}(hour|two hours|90 min|over an hour|forever)', 3.0),
            (r'(wait|waited).{0,10}([3-9]\d|[1-9]\d{2,})\s*(min|minute)', 2.5),  # 30+ minutes
            (r'(took|wait).{0,10}(forever|eternity|ages)', 2.5),
            (r'(waste|wasted).{0,10}(time|hour|evening|lunch)', 3.0),
            (r'(never|didn.t).{0,10}(came|arrived|show|bring).{0,15}(food|order|drink)', 3.0),
        ],
        high_patterns=[
            (r'(wait|waited|waiting).{0,15}(long|too long)', 1.5),
            (r'(slow|forever).{0,10}(service|waiter|server|food)', 2.0),
            (r'(quick|fast|prompt|speedy).{0,10}(service|food|order)', 1.0),
            (r'(reservation|reserv\w*).{0,15}(wait|still|honor)', 1.5),
            (r'(rush|rushed|hurry|hurried)', 1.5),
        ],
    ),

    # -------------------------------------------------------------------------
    # G4: Owner Talent
    # -------------------------------------------------------------------------
    "G4_server": TopicConfig(
        name="G4_server",
        group="G4",
        perspective="Owner Talent",
        description="Server and front-of-house performance",
        critical_patterns=[
            (r'(server|waiter|waitress).{0,20}(rude|terrible|awful|worst|horrible)', 3.0),
            (r'(server|waiter|waitress).{0,20}(amazing|fantastic|best|excellent|outstanding)', 3.0),
            (r'(ignored|neglect|disappear).{0,20}(us|our|table|server|waiter)', 2.5),
            (r'(manager|owner).{0,15}(came|apologiz|compens)', 2.0),
        ],
        high_patterns=[
            # Removed broad pattern - require sentiment qualifier
            (r'(friendly|attentive|helpful|knowledgeable).{0,10}(server|waiter|staff)', 1.5),
            (r'(slow|inattentive|forgetful).{0,10}(server|waiter|service)', 1.5),
            (r'(great|good|excellent|terrible|bad|poor).{0,10}service', 1.5),
            (r'(recommend|suggest)\w*.{0,15}(server|waiter)', 1.0),
            (r'(tip|tipped).{0,10}(well|extra|generously|\d+%)', 1.0),
        ],
    ),

    "G4_kitchen": TopicConfig(
        name="G4_kitchen",
        group="G4",
        perspective="Owner Talent",
        description="Chef and kitchen performance",
        critical_patterns=[
            (r'(chef|cook).{0,20}(came out|table|spoke|met)', 2.5),
            (r'(chef|cook).{0,15}(amazing|incredible|talented|skilled|master)', 3.0),
            (r'(execution|technique|skill).{0,15}(perfect|flawless|impeccable)', 2.5),
            (r'(overcooked|undercooked|burnt|raw).{0,10}(completely|totally|very)', 2.5),
        ],
        high_patterns=[
            # Require quality context - not just mentions
            (r'(chef|cook).{0,15}(prepar|made|creat)', 1.0),
            (r'(prepar\w*|cook\w*|made).{0,15}(perfect|well|beautifully)', 1.5),
            (r'(flavor|season|spice|taste).{0,15}(perfect|well|just right)', 1.5),
            (r'(presentation|plating|plate).{0,15}(beautiful|art|gorgeous)', 1.5),
            (r'(fresh|quality).{0,10}(ingredient|produce|meat|fish)', 1.0),
        ],
    ),

    "G4_environment": TopicConfig(
        name="G4_environment",
        group="G4",
        perspective="Owner Talent",
        description="Ambiance and physical environment",
        critical_patterns=[
            (r'(ambiance|atmosphere|decor).{0,15}(stunning|gorgeous|breathtaking|incredible)', 3.0),
            (r'(ambiance|atmosphere|decor).{0,15}(terrible|awful|horrible|depressing)', 3.0),
            (r'(noise|loud|noisy).{0,15}(couldn.t|hard|impossible).{0,10}(hear|talk|conversation)', 2.5),
            (r'(beautiful|stunning).{0,10}(view|location|setting)', 2.5),
        ],
        high_patterns=[
            (r'(ambiance|atmosphere|decor|interior|design)', 1.5),
            (r'(cozy|warm|inviting|welcoming|comfortable)', 1.0),
            (r'(modern|trendy|stylish|chic|elegant)', 1.0),
            (r'(music|lighting|seating).{0,15}(perfect|nice|great|loud|dim)', 1.0),
            (r'(clean|spotless|immaculate)', 1.0),
        ],
    ),

    # -------------------------------------------------------------------------
    # G5: Owner Operations
    # -------------------------------------------------------------------------
    "G5_capacity": TopicConfig(
        name="G5_capacity",
        group="G5",
        perspective="Owner Operations",
        description="Capacity management and reservations",
        critical_patterns=[
            (r'(reservation|reserv\w*).{0,20}(lost|no record|didn.t have|couldn.t find)', 4.0),
            (r'(overbooked|double.booked)', 3.0),
            (r'(turn\w* away|couldn.t seat|no table)', 2.5),
            (r'(packed|crowded|cramped).{0,15}(couldn.t|barely|impossible)', 2.5),
        ],
        high_patterns=[
            (r'(reservation|reserv\w*|book\w*|table)', 1.0),
            (r'(wait|line|queue).{0,15}(table|seat|hour)', 1.5),
            (r'(walk.in|walk in|no reservation)', 1.5),
            (r'(seat\w*|table).{0,15}(immediately|right away|wait)', 1.0),
            (r'(full|packed|busy|empty|quiet)', 1.0),
        ],
    ),

    "G5_execution": TopicConfig(
        name="G5_execution",
        group="G5",
        perspective="Owner Operations",
        description="Order execution and accuracy",
        critical_patterns=[
            # Require complaint context - avoid "nothing wrong with"
            (r'(got|brought|gave).{0,10}(wrong|incorrect).{0,10}(order|dish|food)', 3.0),
            (r'(order|dish|food).{0,10}(was|came).{0,10}(wrong|incorrect)', 3.0),
            (r'(forgot|forgotten|missing|never).{0,15}(order|dish|appetizer|drink|side)', 3.0),
            (r'(cold|lukewarm|room temperature).{0,15}(food|dish|plate|meal)', 2.5),
            (r'(order|food).{0,15}(completely|totally).{0,10}(wrong|different)', 3.0),
        ],
        high_patterns=[
            (r'(order|food).{0,15}(correct|right|as ordered|perfect)', 1.0),
            (r'(timing|pace|course).{0,15}(perfect|well|off|wrong)', 1.5),
            (r'(coordination|synchronized|together)', 1.5),
            (r'(remember|noted|accommodat).{0,15}(request|special|modification)', 1.0),
            (r'(hot|warm|fresh).{0,15}(food|dish|plate|arrival)', 1.0),
        ],
    ),

    "G5_consistency": TopicConfig(
        name="G5_consistency",
        group="G5",
        perspective="Owner Operations",
        description="Consistency across visits",
        critical_patterns=[
            (r'(used to be|was).{0,15}(better|good|great|amazing).{0,15}(now|recently|anymore)', 3.0),
            (r'(quality|food|service).{0,15}(declined|downhill|worse|dropped)', 3.0),
            (r'(inconsistent|hit.or.miss|hit or miss)', 3.0),
            (r'(every|each).{0,10}(time|visit).{0,15}(different|varies|inconsistent)', 2.5),
        ],
        high_patterns=[
            (r'(always|every time|consistently)', 1.5),
            (r'(first|previous|last).{0,10}(visit|time)', 1.0),
            (r'(regular|come back|return\w*|loyal)', 1.5),
            (r'(same|consistent|reliable).{0,15}(quality|taste|service)', 1.5),
            (r'(been.{0,5}(here|coming)|visit\w*).{0,10}(\d+|many|several|multiple)', 1.5),
        ],
    ),

    # -------------------------------------------------------------------------
    # G6: Owner Strategy
    # -------------------------------------------------------------------------
    "G6_uniqueness": TopicConfig(
        name="G6_uniqueness",
        group="G6",
        perspective="Owner Strategy",
        description="Unique and distinctive offerings",
        critical_patterns=[
            (r'(only|nowhere|first).{0,15}(place|restaurant|spot).{0,15}(find|get|serve|offer)', 3.0),
            (r'(unique|one.of.a.kind|unlike any|never seen)', 3.0),
            (r'(signature|famous|known for|must.try).{0,10}(dish|item)', 2.5),
            (r'(creative|innovative|inventive).{0,15}(menu|dish|chef)', 2.5),
        ],
        high_patterns=[
            (r'(unique|special|different|unusual)', 1.5),
            (r'(creative|innovative|original)', 1.5),
            (r'(signature|specialty|house).{0,10}(dish|item|cocktail)', 1.5),
            (r'(try|must|have to).{0,10}(get|order|try)', 1.0),
            (r'(stand\w* out|set\w* apart|distinguish)', 1.5),
        ],
    ),

    "G6_comparison": TopicConfig(
        name="G6_comparison",
        group="G6",
        perspective="Owner Strategy",
        description="Competitive comparison",
        critical_patterns=[
            (r'(best|better|worse).{0,15}(than|compared|versus).{0,30}(restaurant|place)', 3.0),
            (r'(best|top|number one|\#1).{0,15}(in|around|city|town|area|neighborhood)', 3.0),
            (r'(nothing|no one|nowhere).{0,15}(compare|comes close|beats)', 3.0),
        ],
        high_patterns=[
            # Require restaurant context for comparisons
            (r'(better|worse).{0,10}than.{0,20}(restaurant|place|spot)', 1.5),
            (r'(similar|reminds|like).{0,15}(restaurant|place)', 1.0),
            (r'(best|top|favorite).{0,10}(restaurant|place|spot)', 1.5),
            (r'(other|different).{0,10}(restaurant|place|option)', 1.0),
            (r'(rival|compete|competitor)', 1.5),
        ],
    ),

    "G6_loyalty": TopicConfig(
        name="G6_loyalty",
        group="G6",
        perspective="Owner Strategy",
        description="Customer loyalty and retention",
        critical_patterns=[
            (r'(years|decade).{0,15}(coming|regular|loyal|customer|patron)', 3.0),
            (r'(never|won.t|will not).{0,15}(come back|return|again)', 3.0),
            (r'(lost|lose).{0,15}(customer|patron|business)', 3.0),
            (r'(regular|local|neighborhood).{0,10}(spot|place|go.to|haunt)', 2.5),
        ],
        high_patterns=[
            (r'(come|came|go|went).{0,5}(back|again|return)', 1.5),
            (r'(will|would|definitely).{0,10}(return|come back|recommend)', 1.5),
            (r'(regular|frequent|often|weekly|monthly)', 1.5),
            (r'(loyal|loyalty|fan)', 1.5),
            (r'(first time|new to|just discovered)', 1.0),
        ],
    ),
}

# =============================================================================
# PRE-FILTER KEYWORDS (fast string match before expensive regex)
# =============================================================================

PREFILTER_KEYWORDS = {
    # G1: Customer Health
    "G1_allergy": {'allerg', 'epipen', 'anaphyla', 'reaction', 'cross contam', 'nut free', 'dairy free'},
    "G1_dietary": {'vegetarian', 'vegan', 'gluten', 'kosher', 'halal', 'celiac', 'pescatarian', 'lactose', 'dairy free'},
    "G1_hygiene": {'sick', 'poison', 'roach', 'cockroach', 'mouse', 'rat', 'mice', 'rodent', 'mold', 'rotten',
                   'dirty', 'filthy', 'hair in', 'bug in', 'vomit', 'diarrhea', 'health code', 'undercooked'},

    # G2: Customer Social
    "G2_romance": {'date', 'romantic', 'anniversary', 'honeymoon', 'propos', 'engaged', 'boyfriend', 'girlfriend',
                   'husband', 'wife', 'valentine', 'candlelit', 'intimate'},
    "G2_business": {'business', 'client', 'meeting', 'corporate', 'colleague', 'coworker', 'boss', 'professional',
                    'expense', 'impress'},
    "G2_group": {'party', 'birthday', 'celebration', 'group of', 'graduation', 'shower', 'reunion', 'gathering',
                 'large group', 'private room'},

    # G3: Customer Economic
    "G3_price_worth": {'worth', 'value', 'expensive', 'overpriced', 'rip off', 'ripoff', 'waste', 'budget',
                       'affordable', 'pricey', 'bang for'},
    "G3_hidden_costs": {'hidden', 'surprise', 'fee', 'charge', 'gratuity', 'automatic tip', 'auto tip',
                        'service charge', 'corkage', 'extra charge', 'fine print'},
    "G3_time_value": {'wait', 'slow', 'forever', 'hour', 'minutes', 'took so long', 'never came', 'rushed'},

    # G4: Owner Talent
    "G4_server": {'server', 'waiter', 'waitress', 'attentive', 'ignored', 'neglect', 'rude', 'friendly staff',
                  'great service', 'terrible service'},
    "G4_kitchen": {'chef', 'cook', 'kitchen', 'execution', 'technique', 'plating', 'presentation', 'prepared',
                   'overcooked', 'undercooked', 'burnt', 'seasoned'},
    "G4_environment": {'ambiance', 'atmosphere', 'decor', 'noise', 'loud', 'view', 'cozy', 'lighting', 'modern',
                       'elegant', 'cramped'},

    # G5: Owner Operations
    "G5_capacity": {'reservation', 'overbook', 'wait list', 'no table', 'couldn\'t seat', 'turn away',
                    'walk-in', 'packed', 'empty'},
    "G5_execution": {'wrong order', 'incorrect', 'mistake', 'forgot', 'missing', 'cold food', 'never got',
                     'brought wrong', 'not what i ordered'},
    "G5_consistency": {'inconsistent', 'hit or miss', 'used to be', 'has changed', 'decline', 'downhill',
                       'every time', 'always', 'consistently'},

    # G6: Owner Strategy
    "G6_uniqueness": {'unique', 'only place', 'nowhere else', 'one of a kind', 'signature', 'specialty',
                      'creative', 'innovative', 'original'},
    "G6_comparison": {'better than', 'worse than', 'compared', 'versus', ' vs ', 'best in', 'top', 'favorite',
                      'nothing compares'},
    "G6_loyalty": {'come back', 'return', 'regular', 'loyal', 'years', 'won\'t return', 'never again',
                   'first time', 'frequent'},
}


def compile_patterns(patterns: list[tuple[str, float]]) -> list[tuple[re.Pattern, float]]:
    """Compile regex patterns with case-insensitive flag."""
    return [(re.compile(p, re.IGNORECASE), w) for p, w in patterns]


def passes_prefilter(text_lower: str, keywords: set) -> bool:
    """Fast pre-filter check using simple string matching."""
    return any(kw in text_lower for kw in keywords)


def score_review(text: str, critical: list, high: list, stars: float = None) -> tuple[float, float, list, list]:
    """Score a review against critical and high patterns.

    For incidents (critical), low star ratings (1-2) boost the score.
    For positive mentions (high), high star ratings boost the score.

    Returns: (critical_score, high_score, critical_matches, high_matches)
    """
    critical_score = 0.0
    high_score = 0.0
    critical_matches = []
    high_matches = []

    for pattern, weight in critical:
        if pattern.search(text):
            # Boost critical score for low-star reviews (actual incidents)
            if stars is not None and stars <= 2:
                critical_score += weight * 1.5
            else:
                critical_score += weight
            critical_matches.append(pattern.pattern)

    for pattern, weight in high:
        if pattern.search(text):
            high_score += weight
            high_matches.append(pattern.pattern)

    return critical_score, high_score, critical_matches, high_matches


def process_topic(topic_config: TopicConfig, reviews_path: Path,
                  businesses_path: Path, verbose: bool = True) -> dict:
    """Process a single topic and return restaurant scores."""

    if verbose:
        print(f"\nProcessing {topic_config.name}: {topic_config.description}")

    # Compile patterns
    critical = compile_patterns(topic_config.critical_patterns)
    high = compile_patterns(topic_config.high_patterns)

    # Get pre-filter keywords for fast screening
    prefilter_keywords = PREFILTER_KEYWORDS.get(topic_config.name, set())

    # Track scores per business
    # Use MAX score (peak incident) rather than cumulative
    business_scores = defaultdict(lambda: {
        "max_critical_score": 0.0,
        "max_high_score": 0.0,
        "total_critical_score": 0.0,
        "total_high_score": 0.0,
        "critical_reviews": [],
        "high_reviews": [],
        "total_reviews": 0,
    })

    # Scan all reviews
    review_count = 0
    prefilter_passed = 0
    with open(reviews_path) as f:
        for line in f:
            review_count += 1
            if verbose and review_count % 1_000_000 == 0:
                print(f"  Processed {review_count:,} reviews (pre-filter passed: {prefilter_passed:,})...")

            review = json.loads(line)
            text = review["text"]
            text_lower = text.lower()

            # Fast pre-filter check (skip expensive regex if no keywords match)
            if prefilter_keywords and not passes_prefilter(text_lower, prefilter_keywords):
                continue

            prefilter_passed += 1
            bid = review["business_id"]
            stars = review.get("stars", 3)

            c_score, h_score, c_matches, h_matches = score_review(text, critical, high, stars)

            if c_score > 0 or h_score > 0:
                business_scores[bid]["total_reviews"] += 1

                if c_score > 0:
                    business_scores[bid]["total_critical_score"] += c_score
                    business_scores[bid]["max_critical_score"] = max(
                        business_scores[bid]["max_critical_score"], c_score
                    )
                    business_scores[bid]["critical_reviews"].append({
                        "review_id": review["review_id"],
                        "stars": stars,
                        "date": review["date"],
                        "score": c_score,
                        "matches": c_matches,
                        "snippet": text[:300],
                    })

                if h_score > 0:
                    business_scores[bid]["total_high_score"] += h_score
                    business_scores[bid]["max_high_score"] = max(
                        business_scores[bid]["max_high_score"], h_score
                    )
                    business_scores[bid]["high_reviews"].append({
                        "review_id": review["review_id"],
                        "stars": stars,
                        "date": review["date"],
                        "score": h_score,
                        "matches": h_matches,
                        "snippet": text[:300],
                    })

    if verbose:
        filter_rate = (1 - prefilter_passed / review_count) * 100 if review_count > 0 else 0
        print(f"  Scanned {review_count:,} reviews, pre-filter passed {prefilter_passed:,} ({100-filter_rate:.1f}%)")
        print(f"  Found {len(business_scores)} businesses with matches")

    # Load business metadata
    business_info = {}
    with open(businesses_path) as f:
        for line in f:
            b = json.loads(line)
            if b["business_id"] in business_scores:
                business_info[b["business_id"]] = {
                    "name": b["name"],
                    "city": b.get("city"),
                    "state": b.get("state"),
                    "stars": b.get("stars"),
                    "review_count": b.get("review_count"),
                    "categories": b.get("categories"),
                }

    # Build final results
    results = {
        "topic": topic_config.name,
        "group": topic_config.group,
        "perspective": topic_config.perspective,
        "description": topic_config.description,
        "generated_at": datetime.now().isoformat(),
        "total_reviews_scanned": review_count,
        "businesses_with_matches": len(business_scores),
        "critical_list": [],
        "high_list": [],
    }

    # Sort businesses by score
    for bid, scores in business_scores.items():
        info = business_info.get(bid, {})
        entry = {
            "business_id": bid,
            "name": info.get("name", "Unknown"),
            "city": info.get("city"),
            "state": info.get("state"),
            "stars": info.get("stars"),
            "review_count": info.get("review_count"),
            "categories": info.get("categories"),
            # Use MAX score (peak incident) for categorization
            "max_critical_score": scores["max_critical_score"],
            "max_high_score": scores["max_high_score"],
            "total_critical_score": scores["total_critical_score"],
            "total_high_score": scores["total_high_score"],
            "topic_reviews": scores["total_reviews"],
            "critical_review_count": len(scores["critical_reviews"]),
            "high_review_count": len(scores["high_reviews"]),
            # Keep top 5 reviews as samples (sorted by score)
            "sample_critical_reviews": sorted(
                scores["critical_reviews"],
                key=lambda x: x["score"],
                reverse=True
            )[:5],
            "sample_high_reviews": sorted(
                scores["high_reviews"],
                key=lambda x: x["score"],
                reverse=True
            )[:5],
        }

        # Categorize based on MAX score (peak incident severity)
        # Critical: has at least one review with critical score >= 4 (severe incident)
        # High: has at least one review with critical score >= 2 OR high score >= 3
        if scores["max_critical_score"] >= 4.0:
            results["critical_list"].append(entry)
        elif scores["max_critical_score"] >= 2.0 or scores["max_high_score"] >= 3.0:
            results["high_list"].append(entry)

    # Sort by max critical score, then by total critical score
    results["critical_list"].sort(
        key=lambda x: (x["max_critical_score"], x["total_critical_score"]),
        reverse=True
    )
    results["high_list"].sort(
        key=lambda x: (x["max_high_score"], x["total_high_score"]),
        reverse=True
    )

    if verbose:
        print(f"  Critical list: {len(results['critical_list'])} businesses")
        print(f"  High list: {len(results['high_list'])} businesses")

    return results


def process_and_save_topic(args_tuple):
    """Wrapper for multiprocessing - processes and saves a single topic."""
    name, config, reviews_path, businesses_path, output_dir = args_tuple
    try:
        results = process_topic(config, reviews_path, businesses_path, verbose=True)
        output_file = output_dir / f"{name}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        return name, len(results["critical_list"]), len(results["high_list"]), None
    except Exception as e:
        return name, 0, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Build topic selection from full dataset")
    parser.add_argument("--topic", default=None, help="Process single topic (e.g., G1_allergy)")
    parser.add_argument("--dry-run", action="store_true", help="Show patterns only, don't process")
    parser.add_argument("--data", default="yelp", help="Dataset name")
    parser.add_argument("--parallel", type=int, default=0,
                        help="Number of parallel processes (default=0 means auto-detect CPU cores)")
    args = parser.parse_args()

    # Paths
    raw_dir = Path(f"data/raw/{args.data}")
    reviews_path = raw_dir / f"{args.data}_academic_dataset_review.json"
    businesses_path = raw_dir / f"{args.data}_academic_dataset_business.json"
    output_dir = Path(f"data/topic_selection/{args.data}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths
    if not args.dry_run:
        if not reviews_path.exists():
            print(f"Error: Reviews file not found: {reviews_path}")
            return
        if not businesses_path.exists():
            print(f"Error: Business file not found: {businesses_path}")
            return

    # Select topics to process
    if args.topic:
        if args.topic not in TOPICS:
            print(f"Error: Unknown topic '{args.topic}'")
            print(f"Available topics: {', '.join(TOPICS.keys())}")
            return
        topics_to_process = {args.topic: TOPICS[args.topic]}
    else:
        topics_to_process = TOPICS

    # Dry run: just show patterns
    if args.dry_run:
        for name, config in topics_to_process.items():
            print(f"\n{'='*60}")
            print(f"{name}: {config.description}")
            print(f"Group: {config.group} | Perspective: {config.perspective}")
            print(f"\nCritical patterns ({len(config.critical_patterns)}):")
            for p, w in config.critical_patterns:
                print(f"  [{w:.1f}] {p}")
            print(f"\nHigh patterns ({len(config.high_patterns)}):")
            for p, w in config.high_patterns:
                print(f"  [{w:.1f}] {p}")
        return

    # Process topics
    print(f"Processing {len(topics_to_process)} topics...")
    print(f"Reviews: {reviews_path}")
    print(f"Output: {output_dir}")

    # Determine parallelism: 0 = auto-detect, 1 = sequential, >1 = specified
    n_parallel = args.parallel if args.parallel > 0 else cpu_count()

    if n_parallel > 1 and len(topics_to_process) > 1:
        # Parallel processing
        n_workers = min(n_parallel, len(topics_to_process), cpu_count())
        print(f"Using {n_workers} parallel workers")

        task_args = [
            (name, config, reviews_path, businesses_path, output_dir)
            for name, config in topics_to_process.items()
        ]

        with Pool(n_workers) as pool:
            results = pool.map(process_and_save_topic, task_args)

        # Report results
        print("\n" + "=" * 60)
        print("Summary:")
        for name, critical_count, high_count, error in results:
            if error:
                print(f"  {name}: ERROR - {error}")
            else:
                print(f"  {name}: {critical_count} critical, {high_count} high")
    else:
        # Sequential processing
        for name, config in topics_to_process.items():
            results = process_topic(config, reviews_path, businesses_path)

            # Save results
            output_file = output_dir / f"{name}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved: {output_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
