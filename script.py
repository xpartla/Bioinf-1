from datetime import datetime
import numpy as np
import argparse
from functools import lru_cache

def parse_arguments():
    parser = argparse.ArgumentParser(description="Better emboss water :) ")

    parser.add_argument("-p", action="store_true", help="Use protein scoring matrix (blossum62.txt)")
    parser.add_argument("-d", action="store_true", help="Use DNA scoring matrix (dna.txt)")
    parser.add_argument("-s", metavar="scoreFile", type=str, help="Specify a custom scoring matrix file")
    parser.add_argument("-r", action="store_true", help="use recursive algorithm instead of smith-waterman")

    parser.add_argument("input_files", nargs=2, help="Two FASTA input files with sequences")

    args = parser.parse_args()

    selected_score_options = [args.p, args.d, args.s is not None]
    if sum(selected_score_options) > 1:
        parser.error("You can use only one -p, -d, or -s <file>.")

    if args.p:
        score_file = "blosum62.txt"
    elif args.d:
        score_file = "dna.txt"
    elif args.s:
        score_file = args.s
    else:
        score_file = ''
        parser.error("You must specify a scoring matrix using -p, -d, or -s <file>.")

    first_input, second_input = args.input_files

    return first_input, second_input, score_file, args.r



def readInput(input):
    with open(input, 'r') as file:
        sequence = []
        firstLine = True
        for line in file:
            if firstLine and line.startswith('>'):
                firstLine = False
                continue
            sequence.append(line.strip())
    return ''.join(sequence)

def readMatrix(score):
    with open(score) as f:
        lines = f.readlines()
    scores = {}
    cols = lines[0].split()
    for line in lines[1:len(lines) - 1]:
        items = line.split()
        row = items[0][0]
        row_scores = []
        for item in items[1:]:
            row_scores.append(int(item))
        scores[row] = dict(zip(cols, row_scores))
    return scores

def outputScores(seq1, seq2, scores, gap_open=10, gap_extend=0.5):
    STOP, LEFT, UP, DIAGONAL = 0, 1, 2, 3

    score_matrix = np.zeros((len(seq2) + 1, len(seq1) + 1))
    gap_x = np.zeros((len(seq2) + 1, len(seq1) + 1))
    gap_y = np.zeros((len(seq2) + 1, len(seq1) + 1))
    traceback_matrix = np.zeros((len(seq2) + 1, len(seq1) + 1), dtype=int)

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            gap_x[j, i] = np.maximum(-gap_open + score_matrix[j, i - 1], -gap_extend + gap_x[j, i - 1])
            gap_y[j, i] = np.maximum(-gap_open + score_matrix[j - 1, i], gap_extend + gap_y[j - 1, i])
            score_matrix[j, i] = np.max([
                scores[seq1[i - 1]][seq2[j - 1]] + score_matrix[j - 1, i - 1],
                gap_x[j, i],
                gap_y[j, i],
                0
            ])

            if score_matrix[j, i] == 0:
                traceback_matrix[j, i] = STOP
            elif score_matrix[j, i] == gap_y[j, i]:
                traceback_matrix[j, i] = LEFT
            elif score_matrix[j, i] == gap_x[j, i]:
                traceback_matrix[j, i] = UP
            elif score_matrix[j, i] == scores[seq1[i - 1]][seq2[j - 1]] + score_matrix[j - 1, i - 1]:
                traceback_matrix[j, i] = DIAGONAL

    return gap_x, gap_y, score_matrix, traceback_matrix


def traceback(score_matrix, traceback_matrix, seq1, seq2, STOP=0, LEFT=1, UP=2, DIAGONAL=3):
    aligned_seq1, aligned_seq2, match = [], [], []

    max_value = score_matrix.max()
    max_j, max_i = np.where(score_matrix == max_value)
    max_j, max_i = int(max_j[0]), int(max_i[0])

    post1 = seq1[max_i:] if max_i < len(seq1) else ''
    post2 = seq2[max_j:] if max_j < len(seq2) else ''

    while traceback_matrix[max_j, max_i] != STOP:
        if traceback_matrix[max_j, max_i] == DIAGONAL:
            aligned_seq1.append(seq1[max_i - 1])
            aligned_seq2.append(seq2[max_j - 1])
            match.append("|" if seq1[max_i - 1] == seq2[max_j - 1] else ".")
            max_i -= 1
            max_j -= 1
        elif traceback_matrix[max_j, max_i] == UP:
            aligned_seq1.append(seq1[max_i - 1])
            aligned_seq2.append('-')
            match.append(" ")
            max_i -= 1
        elif traceback_matrix[max_j, max_i] == LEFT:
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[max_j - 1])
            match.append(" ")
            max_j -= 1

    pre1 = seq1[:max_i] if max_i > 0 else ''
    pre2 = seq2[:max_j] if max_j > 0 else ''

    if len(pre1) < len(pre2):
        pre1 = ' ' * (len(pre2) - len(pre1)) + pre1
    elif len(pre2) < len(pre1):
        pre2 = ' ' * (len(pre1) - len(pre2)) + pre2

    aligned_seq1 = pre1 + '(' + ''.join(aligned_seq1[::-1]) + ')' + post1
    aligned_seq2 = pre2 + '(' + ''.join(aligned_seq2[::-1]) + ')' + post2
    match = ' ' * len(pre1) + '(' + ''.join(match[::-1]) + ')'

    return aligned_seq1, aligned_seq2, match

def outputFile(aligned_seq1, aligned_seq2, match, score_matrix, seq1, seq2, runTime, recursive=False):
    with open('output.txt', 'w') as f:
        f.write('#' * 50 + '\n')
        f.write('Rundate: ' + datetime.now().isoformat(timespec='seconds') + '\n')
        f.write('Execution time: ' + str(runTime) + '\n')
        f.write('# seq_1 :' + seq1 + '\n# seq_2: ' + seq2 + '\n')
        f.write('# gapOpen 10 \n')
        f.write('# gapExtend 0.5 \n')
        f.write('#' * 50 + '\n\n')

        f.write('#=================================================\n')
        f.write('# Alignment Score: ' + str(np.max(score_matrix)) + '\n')
        f.write('# Length: ' + str(len(match) - 2) + '\n')
        f.write('#=================================================\n\n')
        f.write(aligned_seq1 + '\n' + match + '\n' + aligned_seq2 + '\n')
        f.write('\n#-------------------------------------------------\n')
        f.write('#-------------------------------------------------\n')
        if recursive:
            f.write('recursion not yet implemented')


def run(firstInput, secondInput, scoreFile, recursive):
    seq1 = readInput(firstInput)
    seq2 = readInput(secondInput)

    scores = readMatrix(scoreFile)
    startTime = datetime.now()
    gap_x, gap_y, score_matrix, traceback_matrix = outputScores(seq1, seq2, scores)

    aligned_seq1, aligned_seq2, match = traceback(score_matrix, traceback_matrix, seq1, seq2)
    endTime = datetime.now()
    runTime = endTime - startTime
    outputFile(aligned_seq1, aligned_seq2, match, score_matrix, seq1, seq2, runTime, recursive)

firstInput, secondInput, scoreFile, recursive = parse_arguments()
run(firstInput, secondInput, scoreFile, recursive)
