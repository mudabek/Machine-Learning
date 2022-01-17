import gym
from gym import spaces
from gym.utils import seeding
import random

def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4 * 3 # 4 for card type and 3 for decks
# Scoring system for counting
points = {1 : -1, 10 : -1, 9: -0.5, 8 : 0, 7 : 0.5, 6 : 1, 5 : 1.5, 4 : 1, 3 : 1, 2 : 0.5}
# Initial expected score from the deck
count_score = 0


def draw_card(np_random):
    # return int(np_random.choice(deck))
    global deck 
    drawn_card = random.choice(deck)
    deck.remove(drawn_card)
    
    global count_score
    count_score = count_score + points[drawn_card]

    # Update deck and score if less than 15 cards left
    if len(deck) < 15:
        deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4 * 3 # 4 for card type and 3 for decks
        count_score = 0

    return drawn_card


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):

    def __init__(self, natural=False, sab=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, count=False):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0

        elif action == 2: # double: deal one card, double the reward and end the game
            self.player.append(draw_card(self.np_random))
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer)) * 2

        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
                
        if count:
            return self._get_obs_with_count(), reward, done, {}
        else:
            return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    
    def _get_obs_with_count(self):
        """ Function the returns state acounting for deck's score """
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), count_score)

    def reset(self, count=False):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        if count:
            return self._get_obs_with_count()
        else:
            return self._get_obs()