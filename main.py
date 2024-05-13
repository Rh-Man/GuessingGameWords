import enum
import tkinter as tkinter
from tkinter import messagebox
import game
import sys
import os

root = tkinter.Tk()
wordleGame = game.WordleGame()

GREEN = "#007d21"
YELLOW = "#e2e600"
BLACK = "#000000"
GREY = "#808080"
WHITE = "#FFFFFF"

root.config(bg=WHITE)
root.title("Wordle")
wordInput = tkinter.Entry(root, highlightbackground = WHITE, highlightthickness=0)
wordInput.grid(row=999, column=0, padx=10, pady=10, columnspan=3)


def init_board():
    for j in range (wordleGame.number_guesses):
        for i in range(wordleGame.word_length):
            label = tkinter.Label(root, text="")
            label.grid(row=j,
                        column=i, padx=20, pady=20)
            label.config(bg=WHITE, fg=BLACK, padx=25, pady=20, borderwidth=0.5, relief="solid")

def playGame():
    guess = wordInput.get().lower()
    wordleGame.guess(guess)
    game_status = wordleGame.get_game_status()
    if wordleGame.get_last_guess():
        guess, squares = wordleGame.get_last_guess()
        for i, letter in enumerate(guess):
            label = tkinter.Label(root, text=letter.upper())
            label.grid(row=wordleGame.turn_number-1,
                       column=i, padx=20, pady=20)
            label.config(bg=squares[i], fg=BLACK, padx=20, pady=20)
        wordInput.delete(0, tkinter.END)
        root.update()
        if game_status == 1:
            MsgBox=messagebox.askquestion('Encore', 'Vous avez gagner ! Voulez vous jouer encore ?')
            if MsgBox == 'yes':
                python = sys.executable
                os.execl(python, python, * sys.argv)
            else:
                messagebox.showinfo('Quitter','Merci pour votre participation!. Bye-bye!')
                root.destroy()

        elif game_status == -1:
            MsgBox=messagebox.askquestion('Encore', 'Vous avez perdu ! le mot etait ' + wordleGame.answer + '. Voulez vous jouer encore ?')
            if MsgBox == 'yes':
                python = sys.executable
                os.execl(python, python, * sys.argv)
            else:
                messagebox.showinfo('Quitter', 'Merci pour votre participation!. Bye-bye!')
                root.destroy()

wordGuessButton = tkinter.Button(root, text="Guess", command=playGame, highlightbackground = WHITE, highlightthickness=0, padx=2, pady=2)
wordGuessButton.grid(row=999, column=3, columnspan=3)
root.bind('<Return>', lambda event: playGame())

init_board()
root.mainloop()