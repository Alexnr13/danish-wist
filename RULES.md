# Danish Wist — House Rules

This document is the single source of truth for the game. The code implements
exactly what is written here; if the two disagree, the code is wrong or this
document needs updating first.

Our game is a house version of *Call-ace Whist* (Danish: *Esmakker Whist*).
Where we differ from published rules, this document wins.

---

## 1. Glossary

| Term | Meaning |
|---|---|
| **Forehand** | The player to the dealer's left. |
| **Cat** | The 3 cards dealt face down to the middle of the table. |
| **Level** | The number of tricks bid, 7–13. |
| **Attachment** | A qualifier added to a bid: *Flip*, *Clubs* or *Halves*. |
| **Declarer** | The player who wins the auction. |
| **Partner** | The holder of the called ace. May be the declarer (see §5). |
| **Declaring side** | Declarer plus partner (or declarer alone). |
| **Defenders** | Everyone not on the declaring side. |
| **Fucdic** | A card the declarer places face down to act as the lowest card of the called suit (§6). |
| **Trick value** | Points per trick for a contract (see §9). |

## 2. Players and cards

- Always 4 players. Play proceeds clockwise.
- A standard 52-card deck plus **3 Jokers** (55 cards).
- Cards rank A K Q J 10 9 8 7 6 5 4 3 2 within each suit.
- Jokers belong to no suit (see §7).

## 3. The deal

- Each player receives 13 cards; the remaining 3 form the **cat**, face down.
  (Three Jokers are what make a 13-card hand plus a 3-card cat possible.)
- The method of dealing does not matter for a well-shuffled deck. Software
  simply shuffles and deals.
- The deal passes clockwise after each hand.
- Play is organised in **sets of four deals**, so every player deals once
  (the auction order gives a small advantage by seat). A session lasts as many
  sets as the players like.

### Iron hand

A player whose hand has **no court cards (J, Q, K), no aces and no Jokers** may
declare *iron hand* before the auction starts. The cards are thrown in and the
**same dealer** deals again.

## 4. The auction

### Bids

A bid is a **level** (7–13), optionally with one **attachment**:

- **Flip** — trumps are chosen by turning cards from the cat (§6).
- **Clubs** — clubs are trumps.
- **Halves** — the partner chooses trumps (§6).

The level is the number of tricks the declaring side promises to take together.

### Which bids beat which

1. A higher level beats a lower level, whatever the attachments.
2. At the same level, any attachment beats a plain bid.
3. At the same level, attachments are **equal in rank; the later one wins**.
   Each attachment may be used **only once per level**.

So after "9 Flip", "9 Clubs" beats it, and then only "9 Halves" or any bid at
level 10 or above can beat that. "9 Flip" cannot be bid again. This allows
tactics such as bidding a less-wanted attachment first, expecting to be
overcalled, and saving your preferred attachment for later.

### Order of bidding

The auction is a series of one-on-one duels:

1. Forehand bids first, or passes.
2. The player to their left must beat the current bid or pass.
3. The two players then alternate, each beating the other's bid or passing,
   until one of them passes.
4. The survivor then duels the next player clockwise in the same way, and so
   on until every other player has had their turn.
5. **A pass is final** — a player who passes takes no further part.
6. The last player standing becomes the **declarer** at their final bid.
7. If all four players pass, the **same dealer** deals again.

The minimum bid is 7. (By custom, 7 and 8 are rarely bid because they are
worth so little, and the last player to speak will usually pass rather than
play a poor hand. This is etiquette, not a rule.)

## 5. Calling a partner

- After winning the auction, the declarer names a suit; whoever holds **the ace
  of that suit** is the partner.
- The partner must not reveal themselves. Partnerships become known only
  when the called ace is played (or, in Halves, when the partner names trumps).
- In **Plain** and **Clubs** contracts, the called ace **may not be in the
  trump suit**. (In Flip and Halves the rules below decide whether they can
  coincide.)
- The declarer plays **alone against three** whenever the called ace is not
  held by another player:
  - the declarer calls an ace **they hold themselves** (a declarer holding all
    four aces has no other option, so such players usually avoid winning the
    auction);
  - the called ace is **in the cat**, whether or not the declarer picks it up
    in the exchange.

  Playing alone changes nothing else about the rules; only the settlement
  differs (§9).

## 6. Trumps and the cat

The order is: call the ace → trumps are decided → exchange with the cat.

### Trumps by contract

| Contract | Who decides trumps |
|---|---|
| Plain | The declarer names any suit except the called ace's suit. |
| Clubs | Clubs are trumps. |
| Halves | The partner names any suit except the called ace's suit, and is thereby revealed. |
| Flip | The cat is turned over (below). |

There are no trumps only when Flip produces a Joker. Spades have no special
status; they are an ordinary suit that can be named as trumps.

### Halves when the declarer is alone

The declarer asks the partner to name trumps. If nobody answers because the
declarer called their own ace, or the called ace is in the cat, the declarer
names trumps instead, still excluding the called ace's suit.

### Flip

After the ace is called, the cat is turned face up **one card at a time**:

- After the first or second card, the declarer may **accept** that card's suit
  as trumps, or turn the next card.
- The third card, if reached, sets trumps and cannot be refused.
- A **Joker** accepted (or reached as the third card) means **no trumps**.
  Flip is the only way a no-trump hand arises.
- Trumps may turn out to be the suit of the called ace.

### Exchanging with the cat

Once trumps are decided, the declarer may exchange with the cat, or decline.
To exchange, the declarer first puts **3 cards from their hand** face down,
and only then picks up **the whole cat**. The discards are chosen before
seeing the cat (apart from any cards already turned face up in Flip). There
is no partial exchange, and no other player may exchange.

The same applies in Flip. The exchange is optional there too, but since
trumps came from a card of the cat, exchanging always gains at least one trump
(or a Joker), so in practice Flip declarers almost always exchange.

### Fucdic

After the exchange and before the first trick, the declarer may declare
**fucdic** if they hold **one or no cards of the called suit** (Jokers do not
count). Fucdic is allowed in every contract.

- With **one** card of the called suit, that card must become the fucdic.
- With **none**, the declarer chooses any card from their hand.
- The chosen card is placed **face down** and stays face down for the whole
  hand. It is never revealed, not even when played.
- From then on it **is the lowest card of the called suit**, ranking below
  the 2 (the "zero"). It follows every rule for a card of that suit: it may be
  led, it must be played when that suit is led and it is the declarer's only
  card of the suit, and it may be thrown away when the declarer cannot follow
  another suit. It wins a trick only if it is led and nobody else plays that
  suit or a trump.
- The declarer announces only that they are taking fucdic, not whether the
  card came from the called suit.

Its main use is to let a declarer with no cards of the called suit lead it,
forcing the partner to play the called ace. Fucdic is almost always taken when
allowed: it tells the partner the declarer is otherwise void in the called
suit, and hides the card's real identity.

## 7. Play

- The **declarer** leads to the first trick.
- Players must **follow suit** if able. A player who cannot follow may play any
  card, including a trump or a Joker.
- The winner of each trick leads to the next.
- Any player may look at the **last trick won** at any time, until the next
  trick is won.

### Who wins a trick

1. If a **Joker was led**, it wins, even if other Jokers are played to the
   trick.
2. Otherwise, the highest trump played wins.
3. Otherwise, the highest card of the suit led wins.

### Jokers

- A Joker may be **led** to any trick, including the first. When a Joker is
  led there is no suit to follow; every other player may play any card.
- A Joker may only be played to a trick led by another card when the player
  **cannot follow suit**. Such a Joker can never win.
- Jokers are never trumps.

### The called ace

- The partner must play the called ace **the first time another player leads
  its suit**.
- If the partner leads that suit themselves, they need not lead the ace, and
  the obligation remains for the next time another player leads the suit.
- This applies equally when the declarer holds the called ace.
- The declarer may lead the called suit, even on the very first trick.

## 8. What each player knows

Some information is hidden, and the software must show each player exactly
what they would know at a real table.

- **Your own hand** is known only to you.
- **The cat** is hidden, except the cards turned face up in Flip, which
  everyone sees. **Whether the declarer exchanged** is public: everyone sees
  the cat picked up. Only the declarer knows which cards were discarded.
- **A fucdic** is known to have been declared, so everyone knows the declarer
  has no other card of the called suit. What the card really is, and whether
  it came from the called suit, is known only to the declarer.
- **The partner** is known only to the holder of the called ace until it is
  revealed to everyone by any of:
  - the called ace being played;
  - the partner naming trumps in Halves (if the declarer names them instead,
    everyone learns the declarer is alone);
  - the called ace being turned up from the cat in Flip.
- **The declarer** knows they are alone at once if they called their own ace,
  or once they pick up the called ace from the cat.
- **Everything else is public**: the auction, the contract, the called suit,
  trumps, every card played to a trick, and who won each trick.

## 9. Scoring

Scoring is **zero-sum**: points won by one side are paid by the other.

### Trick value

The value of each trick depends on the level, doubled if there is an
attachment:

| Level | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|
| Plain | 10 | 20 | 40 | 80 | 160 | 320 | 640 |
| With attachment | 20 | 40 | 80 | 160 | 320 | 640 | 1280 |

Formula: `trick_value = 10 × 2^(level − 7)`, ×2 with an attachment.

### Amount of the contract

- **Made** (tricks taken ≥ level): `amount = tricks_taken × trick_value`.
  Every trick counts, not just the tricks beyond the bid.
- **Failed** (tricks taken < level):
  `amount = (level − tricks_taken) × 2 × trick_value`.
  Only the missing tricks count, but at double value.

Because a made contract scores every trick taken, while a failed one only
pays for the shortfall, overbidding slightly is often worthwhile.

### Settlement

- **Made:** each defender **pays** `amount`.
- **Failed:** each defender **receives** `amount`.
- The declaring side shares the opposite total equally.

| Situation | Each defender | Declarer | Partner |
|---|---|---|---|
| 2 v 2, made | −amount | +amount | +amount |
| 2 v 2, failed | +amount | −amount | −amount |
| Alone (1 v 3), made | −amount | +3 × amount | — |
| Alone (1 v 3), failed | +amount | −3 × amount | — |

Playing alone is therefore effectively worth triple to the declarer, while the
defenders' stakes are unchanged.

### Examples

- Bid **8**, took 10: trick value 20, amount 10 × 20 = **200**.
  In 2 v 2, declarer and partner each gain 200 and each defender loses 200.
- Bid **8**, took 6: trick value 20, amount (8 − 6) × 2 × 20 = **80**.
  In 2 v 2, declarer and partner each lose 80 and each defender gains 80.
- Bid **8 Clubs**, took 8, alone: trick value 40, amount 8 × 40 = 320.
  Declarer gains 960; each defender loses 320.

---

## Future variants (noted, not implemented)

- **Sun contracts** (*Sol*, *Ren sol*, *Bordlægger*…): declarer plays alone
  with no trumps, trying to take one trick or none. Most likely the first
  expansion. Their place in the bidding order and scoring are to be decided.
- **Three-player game**: four hands are still dealt, one unplayed. If a
  Halves declarer's partner is the unplayed hand, the two other players decide
  together what that hand would name as trumps.
- **Calling a king** when holding all four aces.
- **No Joker lead to the first trick.**
- **Iron hand handling:** a per-table setting to redeal automatically, or to
  let the player choose whether to call it.
- Other published attachments: *Strong* (spades trumps), *Sans* (no trumps),
  *Kvarte* (declared solo).
