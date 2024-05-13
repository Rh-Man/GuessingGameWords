from game import WordleGame
import random
import math
from itertools import product
import string
import pandas as pd
from os.path import exists
import copy

class BaseAlgorithm:
    '''
    Cette classe est la base de tous les algorithmes. Elle définit des méthodes et attributs communs à tous les algorithmes.
        word_list : La liste des mots possibles à deviner.
        remaining_word_list : La liste des mots restants à deviner après chaque estimation.
        Verbose : Un indicateur de verbosité qui contrôle la sortie d'informations supplémentaires.
        guesses : Une liste des mots devinés jusqu'à présent.
        bad_letters : Une liste des lettres incorrectes devinées jusqu'à présent.
        good_letters : Une liste des lettres correctes devinées jusqu'à présent.
        right_position : Un dictionnaire qui enregistre les lettres correctes à la bonne position dans le mot deviné.
    '''
    def __init__(self, word_list, Verbose=False) -> None:
        self.word_list = word_list
        self.remaining_word_list = word_list
        self.Verbose = Verbose
        self.guesses = []
        self.bad_letters = []
        self.good_letters = []
        self.right_position = {}

    def reset(self):
        self.guesses = []
        self.bad_letters = []
        self.good_letters = []
        self.right_position = {}

    def create_tree(self, word):

        # TO-DO
        pass

    def make_guess(self) -> str:
        pass

    def make_first_guess(self) -> str:
        guess = random.choice(self.word_list)
        self.guesses.append(guess)
        return guess

    def update_information(self, previous_guess):
        p_guess, squares = previous_guess

        for idx, square in enumerate(squares):
            if square=="GREY":
                self.bad_letters.append(p_guess[idx])
            if square=="YELLOW" or square=="GREEN":
                self.good_letters.append(p_guess[idx])
            if square=="GREEN":
                self.right_position[idx] = p_guess[idx]

        btemp = set()
        for i in self.bad_letters:
            if i not in self.good_letters:
                btemp.add(i)
        self.bad_letters = list(btemp)

    def update_remaining_words(self):
        self.remaining_word_list = [word for word in self.word_list if \
                            all(good_letter in word for good_letter in self.good_letters) \
                            and \
                            all(bad_letter not in word for bad_letter in self.bad_letters) \
                            and \
                            all(word[i]==self.right_position[i] for i in self.right_position.keys()) \
                            and \
                            word not in self.guesses]

class HumanAlgorithm(BaseAlgorithm):
    '''
Cet algorithme simule la stratégie humaine typique, en choisissant un mot aléatoire possible en fonction des retours sur le mot précédent.
    '''
    def __init__(self, word_list) -> None:
        super().__init__(word_list)

    def make_guess(self, previous_guess=None) -> str:
        if previous_guess==None:
            return super().make_first_guess()

        super().update_information(previous_guess)
        super().update_remaining_words()

        guess = random.choice(self.remaining_word_list)
        self.guesses.append(guess)
        return guess


class NaiveFrequencyAlgorithm(BaseAlgorithm):
    '''
Cet algorithme utilise une approche simple basée sur la fréquence des lettres.
Il compte le nombre de fois que chaque lettre apparaît dans les mots restants et choisit le mot qui a le nombre total le plus élevé de ses 5 lettres.
C'est une méthode de sélection naïve qui ne prend pas en compte les interactions entre les lettres.
    '''

    def __init__(self, word_list) -> None:
        super().__init__(word_list)

    def countCharacterFrequency(self) -> dict:
        countDict = {}
        for i in range(ord('a'),ord('z')+1):
            countDict[chr(i)] = 0
        for i in self.remaining_word_list:
            unique_chars = set(list(i.lower().strip()))
            for c in unique_chars:
                countDict[c]+=1
        return countDict

    def make_guess(self,previous_guess=None) -> str:
        if previous_guess==None:
            return super().make_first_guess()

        super().update_information(previous_guess)
        super().update_remaining_words()

        frequency = self.countCharacterFrequency()

        max_aggregate = 0
        guess = ""

        for i in self.remaining_word_list:
            agg_freq = 0
            for j in set(list(i.lower().strip())):
                agg_freq+=frequency[j]
            if(agg_freq>max_aggregate):
                max_aggregate = agg_freq
                guess = i
        self.guesses.append(guess)
        return guess



class MaxEntropyAlgorithm(BaseAlgorithm):
    '''
: Cet algorithme est basé sur la théorie de l'information.
Il choisit le mot avec l'entropie la plus élevée dans la liste des mots disponibles, ce qui réduit le nombre de mots possibles correspondant au motif existant.
    '''

    def __init__(self, word_list) -> None:
        super().__init__(word_list)

    def calculate_entropy(self,word) -> float:
        patterns = {}
        for w in self.remaining_word_list:
            p = ""
            for i in range(len(w)):
                if w[i]==word[i]:
                    p+="g"
                elif w[i] in word:
                    p+="y"
                elif w[i] not in word:
                    p+="b"
            if p in patterns:
                patterns[p]+=1
            else:
                patterns[p]=1

        entropy = 0
        for k,v in patterns.items():
            if v!=0:
                prob = v/len(self.remaining_word_list)
                info = -1 * math.log(prob,2)
                entropy+=prob*info

        return entropy


    def make_guess(self,previous_guess=None) -> str:
        if previous_guess==None:
            return super().make_first_guess()

        super().update_information(previous_guess)
        super().update_remaining_words()

        max_entropy = 0
        guess = random.choice(self.remaining_word_list)

        for word in self.remaining_word_list:
            entropy = self.calculate_entropy(word)
            if(entropy>max_entropy):
                max_entropy = entropy
                guess = word

        self.guesses.append(guess)
        return guess

class GeneticAlgortihm(BaseAlgorithm):
    '''
Cet algorithme utilise une approche génétique pour choisir les mots. 
Il sélectionne les deux mots les plus adaptés parmi les mots restants, les combine de manière aléatoire et mute chaque lettre avec une petite probabilité.
Les mots sont évalués en fonction de leur "fitness", qui peut être déterminée par des facteurs tels que le nombre de lettres correctes ou leur position dans le mot correct.
    '''

    def __init__(self, word_list) -> None:
        super().__init__(word_list)
        self.guesses = {}

    def reset(self):
        super().reset()
        self.guesses = {}

    def update_information(self, previous_guess):
        p_guess, squares = previous_guess
        ind_fitness = 0

        for idx, square in enumerate(squares):
            if square=="GREY":
                self.bad_letters.append(p_guess[idx])
                ind_fitness -= 1
            if square=="YELLOW" or square=="GREEN":
                self.good_letters.append(p_guess[idx])
                ind_fitness += 1
            if square=="GREEN":
                self.right_position[idx] = p_guess[idx]
                ind_fitness -= 1
                ind_fitness += 2

        self.guesses[p_guess] = ind_fitness
        self.guesses = dict(sorted(self.guesses.items(), key=lambda item: item[1], reverse=True))

        btemp = set()
        for i in self.bad_letters:
            if i not in self.good_letters:
                btemp.add(i)
        self.bad_letters = list(btemp)

    def make_first_guess(self) -> str:
        guess = random.choice(self.word_list)
        self.guesses[guess] = None
        return guess

    def check_selections(self, word1, word2) -> bool:
        new_word = ''

        for i in range(len(word1)):
            new_word == word1[:i] + word2[i:]
            if new_word != word1 and new_word != word2 and new_word in self.remaining_word_list:
                return True
        return False

    def make_guess(self, previous_guess=None) -> str:
        if previous_guess==None:
            return self.make_first_guess()

        self.update_information(previous_guess)
        super().update_remaining_words()

        # If only one guess, we want to get a different guess from the first
        if len(self.guesses) < 2:
            guess = random.choice(self.remaining_word_list)

            while guess in self.guesses.keys():
                guess = random.choice(self.remaining_word_list)

            self.guesses[guess] = None
            return guess

        guess = 'not_in_list'

        while guess not in self.remaining_word_list:
            # Select the 2 fittest
            guess1 = None
            guess2 = None

            for word1 in list(self.guesses.keys()):
                double_break = False

                # Only pick the fittest that can create a word in the remainig word list, if none, send a random.
                # This is primarily for speed and was chosen after some testing. Alternatively, we could:
                # [1] turn up the mutation threshhold in general or
                # [2] we could turn up the mutation threshold only if the no new words can be created
                #    i.e. check_selections is False, or
                # [3] let it not converge and return one of the previous picks.

                for word2 in list(self.guesses.keys()):
                    if word1 == word2:
                        continue
                    if self.check_selections(word1, word2):
                        guess1 = word1
                        guess2 = word2
                        double_break = True
                        break

                if double_break: break

            if guess1 is None and guess2 is None:
                guess = random.choice(self.remaining_word_list)

                while guess in self.guesses.keys():
                    guess = random.choice(self.remaining_word_list)

                self.guesses[guess] = None
                return guess

            # Cross over
            inds = random.randint(0,4) # to force crossover - change to (1,3)
            guess = guess1[:inds] + guess2[inds:]

            # Mutate
            for i in range(len(guess)):
                if random.random() < 0.1:
                    letter = random.choice(string.ascii_letters[:27])
                    if i == 0:
                        guess = letter + guess[1:]
                    elif i == len(guess) - 1:
                        guess = guess[:len(guess) - 1] + letter
                    else:
                        guess = guess[:i] + letter + guess[i+1:]

        self.guesses.append(guess)
        return guess

class QLearn(BaseAlgorithm):
    '''
Cet algorithme utilise l'apprentissage par renforcement (Q-learning) pour apprendre à choisir le mot suivant.
Il maintient une politique qui associe chaque état (mot précédemment deviné) à une action (mot suivant à deviner).
L'algorithme met à jour cette politique en fonction des récompenses reçues pour chaque action et des valeurs Q, qui estime la récompense attendue pour chaque paire état-action.
En apprenant à partir des récompenses obtenues pour chaque action, l'algorithme devrait être en mesure de choisir des mots plus efficacement au fil du temps.
    '''
    def __init__(self, word_list, Verbose=False) -> None:
        super().__init__(word_list, Verbose)
        self.policy_or_utilities = {}
        self.counts = [0]*len(word_list)
        file_exists = exists('q_policy.csv')
        if not file_exists:
            self.train_agent()
        else:
            self.policy_or_utilities = pd.read_csv('q_policy.csv')

    def train_agent(self):
        learning_rate = 0.1 # alpha
        lamb = 0.1 # lambda

        for i in range(2000):
            game = WordleGame()
            algo = HumanAlgorithm(word_list=game.get_word_list())

            game_status = game.get_game_status()
            game.guess(algo.make_guess(game.get_last_guess()))
            self.counts[self.word_list.index(game.get_last_guess()[0])] += 1
            while game_status==0:
                # Pick action:
                actions = algo.remaining_word_list
                guess = None
                for a in actions:
                    if a not in self.policy_or_utilities.keys():
                        guess = a
                        break
                if guess is None:
                    count_perc = 1
                    for a in actions:
                        a_count_perc = self.counts[self.word_list.index(a)] / sum(self.counts)
                        if count_perc - a_count_perc >= 0.05:
                            guess = a
                            count_perc = a_count_perc
                if guess is None:
                    max_q_val = -100000
                    for a in actions:
                        if self.policy_or_utilities[a] >= max_q_val:
                            guess = a
                            max_q_val = self.policy_or_utilities[a]

                # Perform action
                game.guess(guess)
                self.counts[self.word_list.index(guess)] += 1
                algo.update_information(game.get_last_guess())
                algo.update_remaining_words()
                game_status = game.get_game_status()

                # Update Q-values
                s = game.guesses[-2]
                s_prime = game.guesses[-1]
                reward = -1
                if game_status == 1: reward = 10

                actions = algo.remaining_word_list
                max_q_val = -1000
                for a in actions:
                    if a in self.policy_or_utilities.keys():
                        if self.policy_or_utilities[a] > max_q_val:
                            max_q_val = self.policy_or_utilities[a]
                if max_q_val == -1000:
                    max_q_val = 0

                q_val_s = 0
                if s in self.policy_or_utilities.keys():
                    q_val_s = self.policy_or_utilities[s]

                self.policy_or_utilities[s] = q_val_s + learning_rate * (-1 + lamb*max_q_val - q_val_s)

            if i % 200 == 0:
                print(f"Percent Complete: {i / 2000 * 100}")
                #print('Q-vals: ', self.policy_or_utilities.items

        print('Training complete!\n')
        self.policy_or_utilities = pd.DataFrame(data=self.policy_or_utilities, index=[0])
        self.policy_or_utilities.to_csv('q_policy.csv')

    def make_guess(self, previous_guess=None) -> str:
        if previous_guess == None:
            return super().make_first_guess()

        super().update_information(previous_guess)
        super().update_remaining_words()

        min_q_val = -1000
        guess = None
        for a in self.remaining_word_list:
            if a in self.policy_or_utilities.keys():
                if self.policy_or_utilities[a][0] >= min_q_val:
                    guess = a
                    min_q_val = self.policy_or_utilities[a][0]

        if guess is not None:
            return guess
        else:
            return super().make_first_guess()

class GreedyDepthAlgorithm(BaseAlgorithm):
    '''
    Pick the words by finding the path to minimal depth which basically minimizes the pool
    of possible words as the algorithm gets feedback.

    Implemenation is inspired by: https://towardsdatascience.com/automatic-wordle-solving-a305954b746e
    '''
    def __init__(self, word_list, Verbose=False) -> None:
        super().__init__(word_list, Verbose)

    def make_guess(self , previous_guess=None) -> str:
        # TO-DO
        # If no previous guesses, make a random guess
        if previous_guess==None:
            return super().make_first_guess()

        return super().make_first_guess()

class GreedyBreadthAlgorithm(BaseAlgorithm):
    '''
    Similar to GreedyDepthAlgorithm but searches the word tree using breadth
    first search. I imagine this will be optimal since we would stop the search
    once a leaf is found (which provides the shortest branch such that we would
    want to pick the child of the root that brings us there).
    '''
    def __init__(self, word_list, Verbose=False) -> None:
        super().__init__(word_list, Verbose)

    def make_guess(self) -> str:
        # TO-DO
        return super().make_first_guess()

if __name__=="__main__":
    # manual tests on algorithms
    game = WordleGame()
    # 1. Human
    #algo = HumanAlgorithm(word_list=game.get_word_list())
    #2. Aggregated Frequency
    #algo = NaiveFrequencyAlgorithm(word_list=game.get_word_list())
    # 4. Greedy Depth
    #algo = GreedyDepthAlgorithm(word_list=game.get_word_list())
    # etc...

    choice = 1
    print("Select an Algorithm:\
          \n1.Human Algorithm\
          \n2.Aggregated Frequency\
          \n3.Entropy Maximization\
          \n4.Genetic Algorithm\
          \n5.Q-Learning")

    choice = int(input())
    if choice == 1:
        algo = HumanAlgorithm(word_list=game.get_word_list())
    elif choice == 2:
        algo = NaiveFrequencyAlgorithm(word_list=game.get_word_list())
    elif choice == 3:
        algo = MaxEntropyAlgorithm(word_list=game.get_word_list())
    elif choice == 4:
        algo = GeneticAlgortihm(word_list=game.get_word_list())
    elif choice == 5:
        algo = QLearn(word_list=game.get_word_list())


    #print("ANSWER:",game.answer)
    game_status = game.get_game_status()
    while game_status==0:
        game.guess(algo.make_guess(game.get_last_guess()))
        game_status = game.get_game_status()
        if game_status==0 and game.get_last_guess():
            guess, squares = game.get_last_guess()
            print('%s: %s'%(guess, ' '.join(squares)))
        elif game_status==1:
            guess, squares = game.get_last_guess()
            print('%s: %s'%(guess, ' '.join(squares)))
            try_again = input("You won! Play another game? (Y/N) ")
            if try_again.upper()=='Y':
                game = WordleGame()
                algo.reset()
                game_status = game.get_game_status()
            elif try_again.upper()=='N':
                print("OK. Bye-bye!")
                break
            else:
                game_status = -2
        elif game_status==-1:
            guess, squares = game.get_last_guess()
            print('%s: %s'%(guess, ' '.join(squares)))
            try_again = input("You lose. Try again? (Y/N) ")
            if try_again.upper()=='Y':
                game.restart()
                algo.reset()
                game_status = game.get_game_status()
            elif try_again.upper()=='N':
                print("OK. Bye-bye!")
                break
            else:
                game_status = -2

        if game_status==-2: # Invalid input
            double_break = False
            while True:
                try_again = input("Input invalid. Try again? (Y/N) ")
                if try_again.upper()=='Y':
                    game.restart()
                    algo.reset()
                    game_status = game.get_game_status()
                    break
                elif try_again.upper()=='N':
                    print("OK. Bye-bye!")
                    double_break = True
                    break
            if double_break:
                break
