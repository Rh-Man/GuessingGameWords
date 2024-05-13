from game import WordleGame
from algorithms import *
import random
from itertools import product
import string
import copy
import time
from IPython.display import clear_output
import pandas as pd

if __name__=="__main__":
    # Get required evaluation test Information
    # First pick algorithms to test
    print("\nWhat algorithm(s) do you want to test?")
    print("If you want multiple, input all corresponding numbers with spaces seperating them.\
    \nFor example, for both Human Algorithm and Aggregated Frequency enter '1 2'.")
    print("Select an Algorithm:\
        \n1.Human Algorithm\
        \n2.Aggregated Frequency\
        \n3.Entropy Maximization\
        \n4.Genetic Algorithm\
        \n5.Q-Learning")
    algos = str(input()).split()
    algos = [int(i) for i in algos]

    # Pick different variables we may want to iterate through
    print('Select number of trials(Default is 100):')
    trials = input()
    if trials == '': trials = 100
    else: trials = int(trials)
    print('Select number of word length(Default is 5):')
    word_length = input()
    if word_length == '': word_length = 5
    else: word_length = int(word_length)
    print('Select number of number of guesses(Default is 6):')
    number_guesses = input()
    if number_guesses == '': number_guesses = 6
    else: number_guesses = int(number_guesses)

    # Loop through n number of times and get performance metrics
    # Calculate win rate, perfect letter accuracy (how many correct letters in correct spots),
    # letter accuracy (how many correct letter), average number of guesses,
    # average number of guesses to win, and time. Subject to change.
    metrics = {}
    for i in range(trials):
        # Defaul params = word_length=5, word_source="default", word_list=None, number_guesses=6
        game = WordleGame(word_length = word_length, number_guesses = number_guesses)

        for alg in algos:
            test_game = copy.deepcopy(game) # Preserve answer

            if alg == 1:
                test_algo = HumanAlgorithm(word_list=test_game.get_word_list())
                test_algo_name = 'Human Algorithm'
            elif alg == 2:
                test_algo = NaiveFrequencyAlgorithm(word_list=test_game.get_word_list())
                test_algo_name = 'Naive Frequency Algorithm'
            elif alg == 3:
                test_algo = MaxEntropyAlgorithm(word_list=test_game.get_word_list())
                test_algo_name = 'Max Entropy Algorithm'
            elif alg == 4:
                test_algo = GeneticAlgortihm(word_list=test_game.get_word_list())
                test_algo_name = 'Genetic Algorithm'
            elif alg == 5:
                test_algo = QLearn(word_list=test_game.get_word_list())
                test_algo_name = 'Q Learning'

            if test_algo_name not in metrics.keys():
                metrics[test_algo_name] = {'Wins' : 0, 'Perfect Letter Count' : 0, 'Correct Letter Count' : 0,
                    'Num Guesses' : 0, 'Win Num Guesses' : 0, 'Times' : []}

            start_time = time.perf_counter()
            game_status = test_game.get_game_status()
            while game_status==0:
                test_game.guess(test_algo.make_guess(test_game.get_last_guess()))
                game_status = test_game.get_game_status()

                if game_status==0 and test_game.get_last_guess():
                    continue
                elif game_status==1:
                    end_time = time.perf_counter()
                    metrics[test_algo_name]['Wins'] += 1
                    metrics[test_algo_name]['Perfect Letter Count'] += word_length
                    metrics[test_algo_name]['Correct Letter Count'] += word_length
                    metrics[test_algo_name]['Num Guesses'] += len(test_algo.guesses)
                    metrics[test_algo_name]['Win Num Guesses'] += len(test_algo.guesses)
                    metrics[test_algo_name]['Times'].append(end_time - start_time)
                    break
                elif game_status==-1:
                    end_time = time.perf_counter()
                    metrics[test_algo_name]['Perfect Letter Count'] += len(test_algo.right_position)
                    metrics[test_algo_name]['Correct Letter Count'] += len(set(test_algo.good_letters))
                    metrics[test_algo_name]['Num Guesses'] += number_guesses
                    metrics[test_algo_name]['Times'].append(end_time - start_time)
                    break

        if i % 10 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    clear_output(wait=True)
    print('Testing complete!\n')

    output_data = {'Algo Name' : [], 'Average Win Rate' : [], 'Letter Accuracy' : [], 'Perfect Letter Accuracy' : [],
        'Average Number of Guesses' : [], 'Average Number of Guesses to Win' : [], 'Average Run Time' : []}

    for i in range(len(metrics)):
        algo = list(metrics.keys())[i]
        algo_values = metrics[algo]

        output_data['Algo Name'].append(algo)
        output_data['Average Win Rate'].append(algo_values['Wins']/trials)
        output_data['Letter Accuracy'].append(algo_values['Correct Letter Count'] / (trials * word_length))
        output_data['Perfect Letter Accuracy'].append(algo_values['Perfect Letter Count'] / (trials * word_length))
        output_data['Average Number of Guesses'].append(algo_values['Num Guesses'] / trials)
        output_data['Average Number of Guesses to Win'].append(algo_values['Win Num Guesses'] / algo_values['Wins'])
        output_data['Average Run Time'].append(sum(algo_values['Times']) / trials)

        print('Average win rate for {} was: '.format(algo), output_data['Average Win Rate'][-1])
        print('Letter accuracy for {} was: '.format(algo), output_data['Letter Accuracy'][-1])
        print('Perfect letter accuracy for {} was: '.format(algo), output_data['Perfect Letter Accuracy'][-1])
        print('Average number of guesses for {} was: '.format(algo), output_data['Average Number of Guesses'][-1])
        print('Average number of guesses to win for {} was: '.format(algo), output_data['Average Number of Guesses to Win'][-1])
        print('Average run time for {} was: '.format(algo), output_data['Average Run Time'][-1])
        print()

    pd.DataFrame(output_data).to_csv('outputs/full_evaluation_out.csv')
