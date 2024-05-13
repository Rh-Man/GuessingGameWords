import requests
import random
import collections
from tkinter import messagebox


class WordleGame:
    def __init__(self, word_length=5, word_source="default", word_list=None, number_guesses=6):
        """
        Initiates the Game class

        Notes:
            - We can create a word list with 3 methods
                1. File source
                    Stored in the `word_lists` directory
                    Current file sources are:
                    i. default_word.txt:  the words that real Wordle uses;
                        note that this source only has words 5 letters long
                2. Web source
                    This should be dealt with in here since the formats vary
                    Current web sources are:
                    i. web_simple: this is a list created by MIT; this is a
                        strong option when varying word_length
                3. Manual
                    The user must input the list in the `word_list` parameter

        Parameters
        ----------
        word_length: int, optional
            Length of word to guess
        word_source: str, optional
            Where to get the input of the words
        word_list: list, optional
            If word_source is 'manual', then this is the list of allowed word guesses
        number_guesses: int, optional
            Max number of times a word can be guessed

        Returns
        None
        -------
        """
        # check parameters are correct
        if word_source not in ["default", "web_simple", "manual"]:
            raise ValueError(
                "word_list parameter is 'default', 'word_simple' or 'manual'.")
        if not isinstance(word_length, int):
            raise ValueError("word_length must be an int.")
        if not isinstance(number_guesses, int):
            raise ValueError("number_guesses must be an int.")
        # set up variables
        if word_source == 'default':
            words = open('word_lists/default_words.txt', 'r')
            word_list = [word for word in words.read().splitlines()]
            words.close()
        elif word_source == "manual":
            if word_list == None:
                raise ValueError("word_list must be provided.")
            if not type(word_list) is list and not type(word_list) is tuple:
                raise ValueError("word_list must be in list or tuple form.")
            if not all([isinstance(word, str) for word in word_list]):
                raise ValueError("all words in word_list must be strings.")
        elif word_source == "web_simple":
            url = "https://www.mit.edu/~ecprice/wordlist.10000"
            r = requests.get(url, stream=True)
            word_list = [word for word in r.text.split(
                "\n") if len(word) == word_length]
        self.word_list = word_list
        self.answer = random.choice(self.word_list)
        self.turn_number = 0
        self.number_guesses = number_guesses
        self.guesses = []
        self.WIN = 1
        self.LOSE = -1
        self.game_status = 0
        self.word_length = word_length

    def get_guesses(self):
        '''
        Gives previous guesses and the square color of each letter,
        which indicates the correctness of the guess.

        Parameters
        ----------
        None

        Returns
        -------
        guesses: tuple
            The guesses structured as (guess, colors of letter blocks)
        '''
        squares = []
        for guess in self.guesses:
            square = []
            letter_counter = collections.Counter(s for s, g in zip(self.answer, guess) if s != g)
            for i, letter in enumerate(guess):
                if letter == self.answer[i]:
                    square.append("GREEN")
                elif letter in self.answer and letter_counter[letter] > 0:
                    square.append("YELLOW")
                    letter_counter[letter] -= 1
                else:
                    square.append("GREY")
            squares.append(square)
        guesses = [list(x) for x in zip(self.guesses, squares)]
        return guesses

    def get_last_guess(self):
        '''
        Get most recent word guess

        Parameters
        ----------
        None

        Returns
        -------
        last_guess: str
            Last word guesses
        last_squares: list
            The color of the square of the last guess
        '''
        guesses = self.get_guesses()
        if not guesses:
            return None
        last_guess, last_squares = guesses[-1]
        return last_guess, last_squares

    def get_game_status(self):
        return self.game_status

    def get_word_list(self):
        return self.word_list

    def restart(self):
        '''
        Restarts game to default

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.turn_number = 0
        self.guesses = []
        self.game_status = 0

    def guess(self, guess):
        '''
        Guess the answer for the game

        Parameters
        ----------
        guess: string
            The guess of the word; must be same length
            and in the word list

        Returns
        -------
        game_status: int
            0 if in progress, -1 for lose, 1 for win
        '''
        if not len(guess) == self.word_length:
            messagebox.showinfo(
                "Error!", f'Must guess a word of length %d' % self.word_length)
            return self.game_status
        if not guess in self.word_list:
            messagebox.showinfo(
                "Error!", f'Le mot doit se trouver dans la liste')
            return self.game_status
        if not guess in self.word_list:
            # raise ValueError('Word is not valid. Must be in the word list.')
            messagebox.showinfo(
                "Error!", f'Le mot doit se trouver dans la liste')
            return self.game_status
        self.guesses.append(guess)
        self.turn_number += 1
        if guess == self.answer:
            self.game_status = self.WIN
            return self.game_status
        if not self.turn_number<self.number_guesses:
            self.game_status = self.LOSE
            return self.game_status
        return self.game_status


if __name__ == "__main__":
    # test manual word list input
    '''
    words = open('word_lists/default_words.txt', 'r')
    word_list = [word for word in words.read().splitlines()]
    words.close()
    game = WordleGame(word_source="manual", word_list=word_list)
    '''
    # test web source list input
    '''
    game = WordleGame(word_source="web_simple")
    '''
    # test with default word list
    game = WordleGame()
    game_status = game.get_game_status()
    print(game.answer)
    while game_status == 0:
        guess = input("Enter guess: ")
        game.guess(guess)
        game_status = game.get_game_status()
        if game_status == 0 and game.get_last_guess():
            guess, squares = game.get_last_guess()
            print('%s: %s' % (guess, ' '.join(squares)))
        elif game_status == 1:
            guess, squares = game.get_last_guess()
            print('%s: %s'%(guess, ' '.join(squares)))
            try_again = input("You won! Play another game? (Y/N) ")
            if try_again == 'Y':
                game = WordleGame()
                game_status = game.get_game_status()
            else:
                print("OK. Bye-bye!")
                break
        elif game_status == -1:
            guess, squares = game.get_last_guess()
            print('%s: %s'%(guess, ' '.join(squares)))
            try_again = input("You lose. Try again? (Y/N) ")
            if try_again == 'Y':
                game.restart()
                game_status = game.get_game_status()
            else:
                print("OK. Bye-bye!")
                break
