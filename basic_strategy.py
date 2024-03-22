class BasicStrategy:
    def __init__(self, player_hand, dealer_hand):
        self.player_hand = sorted(player_hand, reverse = True)
        self.dealer_hand = sorted(dealer_hand, reverse = True)

    def recommend(self):
        # Determine whether to hit, stand, double, or split based on basic strategy.
        # hands will contain 2-10 and A.    J, Q, K already converted to 10
        move = ""
        if len(self.player_hand) < 2:
            raise ValueError
        dealer = self.dealer_hand[0]

        # check blackjack win condition
        if self.player_hand == ["A", "10"]:
            return "Blackjack, you win!"

        # check pairs:
        if len(set(self.player_hand)) == 1:
            card = self.player_hand[0]
            move = "Don't split."
            # always split on A/8 pairs
            if card in ("A", "8"):
                move = "Split."
            # never split on 5/10 value pair
            elif card in ("5", "10"):
                move = "Don't split."
            # never split on pairs 2 - 7 if dealer btw 8 - A
            elif 2 <= int(card) <= 7:
                if dealer in ("8", "9", "10", "A"):
                    move = "Don't split."
                # split for dealer 2-7 for pairs 2, 3, and 7
                elif card in ("2", "3", "7") and 2 <= int(dealer) <= 7:
                    move = "Split."
                # double down split on dealer 2 and 3 for 2 a 3 pairs
                if card in ("2", "3") and dealer in ("2", "3"):
                    move = "Double Down Split. If not possible, then hit."
                # For 6 pair, split from dealer 3-6, double down on dealer 2
                if card == "6":
                    if dealer.isdigit() and 3 <= int(dealer) <= 6: move = "Split."
                    elif dealer == "2": move = "Double Down Split. If not possible, then hit."
                # For 4 pair, double down on dealer 5 and 6. Don't split on dealer 2-4 and 7.
                if card == "4":
                    if dealer in ("5", "6"): move = "Double Down Split. If not possible, then hit."
            # last case, 9 pair: split on dealer 2-6, 8, 9. Don't split on 7, 10, A.
            else:
                if dealer not in ("7", "10", "A"): move = "Split."

        # soft totals (contains an A). We're reverse sorting the list of cards
        # so A will be the first element.
        elif self.player_hand[0] == 'A':
            second = self.player_hand[1]
            move = "Hit." # default move
            # for soft total 19 or 20, usually stand. Double on soft 19 with dealer 6.
            if second in ("8", "9"):
                if second == "8" and dealer == "6":
                    move = "Double."
                else: move = "Stand."
            elif second == "7":
                if dealer.isdigit() and 2 <= int(dealer) <= 6:
                    move = "Double."
                elif dealer.isdigit() and 7 <= int(dealer) <= 8:
                    move = "Stand."
            else:
                if dealer in ("5", "6"):
                    move = "Double."
                elif dealer == "4" and second in ("4", "5", "6"):
                    move = "Double."
                elif dealer == "3" and second == "6":
                    move = "Double."

        # hard totals, no A.
        else:
            hard_total = self.calculate_hand_total()
            move = "Hit." # default move
            if hard_total > 17:
                move = "Stand."
            elif hard_total == 17:
                move = "Stand."
            elif 13 <= hard_total <= 16 and dealer.isdigit() and 2 <= int(dealer) <= 6:
                move = "Stand."
            elif hard_total == 12 and dealer.isdigit() and 4 <= int (dealer) <= 6:
                move = "Stand."
            elif hard_total == 11:
                move = "Double."
            elif hard_total == 10 and dealer.isdigit() and 2 <= int(dealer) <= 9:
                move = "Double."
            elif hard_total == 9 and dealer.isdigit() and 3 <= int(dealer) <= 6:
                move = "Double."

        #print(move)
        return move

    def calculate_hand_total(self):
        # Calculate the total 'hard' value of a blackjack hand.
        total = 0
        num_aces = 0
        cards = self.player_hand
        cards.sort(reverse=False)
        for card in cards:
            if card.isdigit():
                total += int(card)
            elif card == 'A':
                num_aces += 1
                total += 11

        # Adjust total for aces if needed
        while total > 21 and num_aces > 0:
            total -= 10
            num_aces -= 1

        return total


"""
# testing for all possible hands

player_hands = [['2', '2'], ['3', '3'], ['4', '4'], ['5', '5'], ['6', '6'], ['7', '7'], ['8', '8'], ['9', '9'], ['10', '10'], ['A', 'A'], ['2', '3'], ['2', '4'], ['2', '5'], ['2', '6'], ['2', '7'], ['2', '8'], ['2', '9'], ['2', '10'], ['2', 'A'], ['3', '4'], ['3', '5'], ['3', '6'], ['3', '7'], ['3', '8'], ['3', '9'], ['3', '10'], ['3', 'A'], ['4', '5'], ['4', '6'], ['4', '7'], ['4', '8'], ['4', '9'], ['4', '10'], ['4', 'A'], ['5', '6'], ['5', '7'], ['5', '8'], ['5', '9'], ['5', '10'], ['5', 'A'], ['6', '7'], ['6', '8'], ['6', '9'], ['6', '10'], ['6', 'A'], ['7', '8'], ['7', '9'], ['7', '10'], ['7', 'A'], ['8', '9'], ['8', '10'], ['8', 'A'], ['9', '10'], ['9', 'A'], ['10', 'A'], ['A', '2'], ['A', '3'], ['A', '4'], ['A', '5'], ['A', '6'], ['A', '7'], ['A', '8'], ['A', '9'], ['A', '10']]
dealer_upcards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
for player_hand in player_hands:
    print()
    for dealer_upcard in dealer_upcards:
        strategy = BasicStrategy(player_hand, [dealer_upcard])
        move = strategy.recommend()
        print("Player Hand:", player_hand, "| Dealer Upcard:", dealer_upcard, "| Recommended Move:", move)

#strategy = BasicStrategy(player_hand, dealer_hand)
#move = strategy.recommend()
#print("Recommended move:", move)
"""