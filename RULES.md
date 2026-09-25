# Danish Wist — House Rules

This document is the single source of truth for the game. The code implements
exactly what is written here; if the two disagree, the code is wrong or this
document needs updating first.

Our game is a house version of *Call-ace Whist* (Danish: *Esmakker Whist*).
Where we differ from published rules, this document wins.

Items marked **[Qn]** are open questions, listed at the end.

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
| **Trick value** | Points per trick for a contract (see §8). |

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
declare *iron hand*, and the cards are thrown in and redealt. **[Q1]**

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
7. If all four players pass, the hand is redealt. **[Q2]**

## 5. Calling a partner

- After winning the auction, the declarer names a suit; whoever holds **the ace
  of that suit** is the partner.
- The partner must not reveal themselves. Partnerships become known only
  when the called ace is played (or, in Halves, when the partner names trumps).
- The declarer may call an ace **they hold themselves**, in which case they
  play **alone against three**. A declarer holding all four aces has no other
  option. (Players holding all four aces will therefore usually avoid winning
  the auction.)
- **[Q3]** May the called ace be in the trump suit?
- **[Q4]** What if the called ace is in the cat?

## 6. Trumps and the cat

The order is: call the ace → trumps are decided → exchange with the cat.

### Trumps by contract

| Contract | Who decides trumps |
|---|---|
| Plain | The declarer names any suit. |
| Clubs | Clubs are trumps. |
| Halves | The partner names trumps, and is thereby revealed. **[Q5]** |
| Flip | The cat is turned over (below). |

### Flip

After the ace is called, the cat is turned face up **one card at a time**:

- After the first or second card, the declarer may **accept** that card's suit
  as trumps, or turn the next card.
- The third card, if reached, sets trumps and cannot be refused.
- A **Joker** accepted (or reached as the third card) means **no trumps**.
  Flip is the only way a no-trump hand arises.
- Trumps may turn out to be the suit of the called ace.

### Exchanging with the cat

The declarer may exchange cards with the cat, or decline. **[Q6]**

## 7. Play

- **[Q7]** Who leads to the first trick?
- Players must **follow suit** if able. A player who cannot follow may play any
  card, including a trump or a Joker.
- The winner of each trick leads to the next.

### Who wins a trick

1. If a **Joker was led**, it wins. **[Q8]**
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
  its suit**. **[Q9]**
- The partner is not obliged to lead the ace if they lead that suit
  themselves.
- The declarer may lead the called suit, even on the very first trick.

## 8. Scoring

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
- **Three-player game** with a dummy hand.
- **Calling a king** when holding all four aces.
- **No Joker lead to the first trick.**
- **Iron hand handling:** a per-table setting to redeal automatically, or to
  let the player choose whether to call it.
- Other published attachments: *Strong* (spades trumps), *Sans* (no trumps),
  *Kvarte* (declared solo). **[Q10]**

## Open questions

- **Q1** Is iron hand declared before the auction starts? Can any player
  declare it, or only when it's their turn to bid? Does the same dealer redeal?
- **Q2** After four passes, does the same dealer redeal, or does the deal
  move on?
- **Q3** In a Plain or Clubs contract, may the declarer call the ace of the
  trump suit? (For Plain, the order "call the ace, then name trumps" matters.)
- **Q4** If the called ace is in the cat, does the declarer play alone? What
  if the declarer takes the called ace from the cat during the exchange?
- **Q5** In Halves, may the partner choose the suit of the called ace? What
  happens if the declarer called their own ace, or the ace is in the cat?
- **Q6** Must the exchange be all 3 cards or none, or may the declarer swap
  1 or 2? If the declarer declines, can another player exchange instead? In
  Flip, where the cat is face up, is the exchange the same?
- **Q7** You said forehand leads first, and later that the declarer leads.
  Which is it?
- **Q8** If a Joker is led and other Jokers follow, the first (led) Joker
  wins — correct?
- **Q9** If the partner leads the called suit with a small card, is the ace
  still due the next time someone else leads that suit, or has the obligation
  lapsed?
- **Q10** You mentioned playing "spades". Is *Strong* (spades trumps) a fourth
  attachment we use, or was that a slip?
